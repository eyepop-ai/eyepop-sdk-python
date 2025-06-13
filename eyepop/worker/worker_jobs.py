import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from typing import Callable, BinaryIO, Any
from pydantic import TypeAdapter

import aiohttp

from eyepop.worker.worker_client_session import (WorkerClientSession)
from eyepop.jobs import Job, JobStateCallback
from eyepop.worker.worker_types import VideoMode, ComponentParams

log_requests = logging.getLogger('eyepop.requests')


class WorkerJob(Job):
    """
    Abstract Job submitted to an EyePop.ai WorkerEndpoint.
    """
    _component_params = list[ComponentParams] | None
    def __init__(self,
                 session: WorkerClientSession,
                 component_params: list[ComponentParams] | None,
                 on_ready: Callable[["WorkerJob"], None] | None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self._component_params = component_params

    async def predict(self) -> dict:
        result = await self.pop_result()
        if result is not None:
            event = result.get('event', None)
            if event is not None:
                event_type = event.get('type', None)
                if event_type == 'error':
                    source_id = event.get('source_id', None)
                    message = event.get('message', None)
                    raise ValueError(f"Error in source {source_id}: {message}")
        return result

    async def _do_read_response(self, queue: Queue) -> bool:
        got_results = False
        if self._response is not None:
            response = self._response
            try:
                self._callback.first_result(self)
                while line := await response.content.readline():
                    # TODO aiohttp should do do this internally
                    for trace in response._traces:
                        await trace.send_response_chunk_received(
                            response.method, response.url, line
                        )
                    if not got_results:
                        got_results = True
                    prediction = json.loads(line)
                    await self.push_result(prediction)
            finally:
                response.close()
        return got_results


class _UploadJob(WorkerJob):
    mime_type: str
    video_mode: VideoMode | None
    open_stream: Callable[[], Any]
    needs_full_duplex: bool

    def __init__(self,
                 mime_type: str,
                 open_stream: Callable[[], Any],
                 video_mode: VideoMode | None,
                 component_params: list[ComponentParams] | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None
     ):
        super().__init__(
            session=session,
            component_params=component_params,
            on_ready=on_ready,
            callback=callback
        )
        self.mime_type = mime_type
        self.video_mode = video_mode
        self.open_stream = open_stream
        self.needs_full_duplex = self.mime_type.startswith("video/")

    def open_mp_writer(self):
        mp_writer = aiohttp.MultipartWriter('form-data')
        if self._component_params is not None:
            component_params_part = mp_writer.append_json(
                TypeAdapter(list[ComponentParams]).dump_python(self._component_params))
            component_params_part.set_content_disposition(
                'form-data', name='params', filename='blob')
        file_part = mp_writer.append(self.open_stream(), {'Content-Type': self.mime_type})
        file_part.set_content_disposition('form-data', name='file', filename='blob')
        return mp_writer


    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        got_result = False
        try:
            if self.needs_full_duplex:
                self._response = await session.pipeline_post(
                    'prepareSource?timeout=120s',
                    accept='application/jsonl',
                    timeout=aiohttp.ClientTimeout(total=None, sock_read=60)
                )
                # Read first event to get the prepared source id and then use
                # two HTTP connections to simulate ful duplex HTTP.
                source_id = None
                line = await self._response.content.readline()
                if line:
                    json_response = json.loads(line)
                    event = json_response.get('event', None)
                    if event is not None:
                        if event.get('type', None) == 'prepared':
                            source_id = event.get('source_id', None)
                if source_id is None:
                    raise ValueError("did not get a prepared sourceId to simulate full duplex")
                video_mode_query = f'&videoMode={self.video_mode.value}' if self.video_mode else ''
                upload_url = f'source?mode=queue&processing=async&sourceId={source_id}{video_mode_query}'
                if self._component_params is None:
                    upload_coro = session.pipeline_post(upload_url,
                                                        accept='application/jsonl',
                                                        open_data=self.open_stream,
                                                        content_type=self.mime_type,
                                                        timeout=aiohttp.ClientTimeout(total=None, sock_read=60))
                else:
                    upload_coro = session.pipeline_post(upload_url,
                                                        accept='application/jsonl',
                                                        open_data=self.open_mp_writer,
                                                        timeout=aiohttp.ClientTimeout(total=None, sock_read=60))
                read_coro = self._do_read_response(queue)
                _, got_result = await asyncio.gather(upload_coro, read_coro)
            else:
                upload_url = f'source?mode=queue&processing=sync'
                if self._component_params is None:
                    self._response = await session.pipeline_post(upload_url,
                                                                 accept='application/jsonl',
                                                                 open_data=self.open_stream,
                                                                 content_type=self.mime_type,
                                                                 timeout=aiohttp.ClientTimeout(total=None, sock_read=60))
                else:
                    self._response = await session.pipeline_post(upload_url,
                                                                 accept='application/jsonl',
                                                                 open_data=self.open_mp_writer,
                                                                 timeout=aiohttp.ClientTimeout(total=None, sock_read=60))
                got_result = await self._do_read_response(queue)
        finally:
            if not got_result:
                pass
                # await queue.put(None)

def _guess_mime_type_from_location(location: str):
    mime_types = mimetypes.guess_type(location)
    if len(mime_types) > 0 and mime_types[0] is not None:
        mime_type = mime_types[0]
    else:
        mime_type = 'application/octet-stream'
    return mime_type


class _UploadFileJob(_UploadJob):
    def __init__(self,
                 location: str,
                 video_mode: VideoMode | None,
                 component_params: list[ComponentParams] | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None
    ):
        super().__init__(mime_type=_guess_mime_type_from_location(location),
                         open_stream=self.open_stream,
                         video_mode=video_mode,
                         component_params=component_params,
                         session=session,
                         on_ready=on_ready,
                         callback=callback
        )
        self.location = location

    def open_stream(self):
        return open(self.location, 'rb')


class _UploadStreamJob(_UploadJob):
    def __init__(self,
                 stream: BinaryIO,
                 mime_type: str,
                 video_mode: VideoMode | None,
                 component_params: list[ComponentParams] | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None
    ):
        super().__init__(mime_type=mime_type,
                         open_stream=self.open_stream,
                         video_mode=video_mode,
                         component_params=component_params,
                         session=session,
                         on_ready=on_ready,
                         callback=callback
        )
        self.stream = stream

    def open_stream(self):
        return self.stream


class _LoadFromJob(WorkerJob):
    def __init__(self,
                 location: str,
                 component_params: list[ComponentParams] | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None
     ):
        super().__init__(
            session=session,
            component_params=component_params,
            on_ready=on_ready,
            callback=callback
        )
        self.location = location
        self.target_url = 'source?mode=queue&processing=sync'
        self.body = {
            "sourceType": "URL",
            "url": self.location,
        }
        if self._component_params is not None:
            self.body['params'] = TypeAdapter(list[ComponentParams]).dump_python(self._component_params)

        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=60)

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        try:
            got_result = False
            self._response = await session.pipeline_patch(self.target_url,
                                                          accept='application/jsonl',
                                                          data=json.dumps(self.body),
                                                          content_type='application/json',
                                                          timeout=self.timeouts)
            got_result = await self._do_read_response(queue)
        finally:
            if not got_result:
                await queue.put(None)


class _LoadFromAssetUuidJob(WorkerJob):
    def __init__(self,
                 asset_uuid: str,
                 component_params: list[ComponentParams] | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None
     ):
        super().__init__(
            session=session,
            component_params=component_params,
            on_ready=on_ready,
            callback=callback
        )
        self.asset_uuid = asset_uuid
        self.target_url = 'source?mode=queue&processing=sync'
        self.body = {
            "sourceType": "ASSET_UUID",
            "assetUuid": self.asset_uuid,
        }
        if self._component_params is not None:
            self.body['params'] = TypeAdapter(list[ComponentParams]).dump_python(self._component_params)

        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=60)

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        try:
            got_result = False
            self._response = await session.pipeline_patch(self.target_url,
                                                          accept='application/jsonl',
                                                          data=json.dumps(self.body),
                                                          content_type='application/json',
                                                          timeout=self.timeouts)
            got_result = await self._do_read_response(queue)
        finally:
            if not got_result:
                await queue.put(None)

