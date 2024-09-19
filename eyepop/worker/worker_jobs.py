import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from typing import Callable, BinaryIO, final

import aiohttp

from eyepop.worker.worker_client_session import (WorkerClientSession)
from eyepop.jobs import Job, JobStateCallback

log_requests = logging.getLogger('eyepop.requests')


class WorkerJob(Job):
    """
    Abstract Job submitted to an EyePop.ai WorkerEndpoint.
    """

    def __init__(self,
                 session: WorkerClientSession,
                 params: dict | None,
                 on_ready: Callable[["WorkerJob"], None] | None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self._params = params


    async def predict(self) -> dict:
        return await self.pop_result()

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
    def __init__(self,
                 mime_type: str,
                 open_stream: Callable[[], any],
                 params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.mime_type = mime_type
        self.open_stream = open_stream
        self.needs_full_duplex = self.mime_type.startswith("video/")

    def open_mp_writer(self):
        mp_writer = aiohttp.MultipartWriter('form-data')
        params_part = mp_writer.append_json(self._params)
        params_part.set_content_disposition('form-data', name='params', filename='blob')
        file_part = mp_writer.append(self.open_stream(), {'Content-Type': self.mime_type})
        file_part.set_content_disposition('form-data', name='file', filename='blob')

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
                upload_url = f'source?mode=queue&processing=async&sourceId={source_id}'
                if self._params is None:
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
                if self._params is None:
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
    if len(mime_types) > 0:
        mime_type = mime_types[0]
    else:
        mime_type = 'application/octet-stream'
    return mime_type


class _UploadFileJob(_UploadJob):
    def __init__(self, location: str, params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(_guess_mime_type_from_location(location),
                         self.open_stream,
                         params,
                         session, on_ready, callback)
        self.location = location

    def open_stream(self):
        return open(self.location, 'rb')


class _UploadStreamJob(_UploadJob):
    def __init__(self, stream: BinaryIO, mime_type: str, params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(mime_type,
                         self.open_stream,
                         params,
                         session, on_ready, callback)
        self.stream = stream

    def open_stream(self):
        return self.stream


class _LoadFromJob(WorkerJob):
    def __init__(self, location: str, params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.location = location
        self.target_url = 'source?mode=queue&processing=sync'
        self.body = {
            "sourceType": "URL",
            "url": self.location,
        }
        if self._params is not None:
            self.body['params'] = self._params

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

