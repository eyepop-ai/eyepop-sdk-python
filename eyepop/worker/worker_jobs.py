import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from typing import Any, BinaryIO, Callable
from urllib.parse import urlencode

import aiohttp
from pydantic import TypeAdapter

from eyepop.data.types.asset import Area
from eyepop.jobs import Job, JobStateCallback
from eyepop.worker.worker_client_session import WorkerClientSession
from eyepop.worker.worker_types import (
    DEFAULT_PREDICTION_VERSION,
    ComponentParams,
    MotionDetectConfig,
    PredictionVersion,
    VideoMode,
)

log_requests = logging.getLogger('eyepop.requests')


class WorkerJob(Job):
    """Abstract Job submitted to an EyePop.ai WorkerEndpoint."""
    _component_params: list[ComponentParams] | None
    _motion_detect: MotionDetectConfig | None
    _roi: Area | None
    _fps: str | None
    _version: PredictionVersion
    _media_cache_seconds: int | None

    def __init__(
            self,
            session: WorkerClientSession,
            component_params: list[ComponentParams] | None,
            motion_detect: MotionDetectConfig | None,
            roi: Area | None,
            fps: str | None,
            media_cache_seconds: int | None,
            on_ready: Callable[["WorkerJob"], None] | None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        super().__init__(session, on_ready, callback)
        self._component_params = component_params
        self._motion_detect = motion_detect
        self._roi = roi
        self._fps = fps
        self._media_cache_seconds = media_cache_seconds
        self._version = version

    async def predict(self) -> dict[str, Any] | None:
        while True:
            result = await self.pop_result()
            if result is None:
                return None
            else:
                event = result.get('event', None)
                if event is not None:
                    if event == 'error':
                        source_id = result.get('source_id', None)
                        message = result.get('message', None)
                        raise ValueError(f"Error in source {source_id}: {message}")
                    type_ = event.get('type', None)
                    if type_ == 'error':
                        source_id = event.get('source_id', None)
                        message = event.get('message', None)
                        raise ValueError(f"Error in source {source_id}: {message}")
                else:
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
                    await self.push_message(prediction)
            finally:
                response.close()
        return got_results


class _UploadJob(WorkerJob):
    mime_type: str
    video_mode: VideoMode | None
    open_stream: Callable[[], Any]
    needs_full_duplex: bool

    def __init__(
            self,
            mime_type: str,
            open_stream: Callable[[], Any],
            video_mode: VideoMode | None,
            component_params: list[ComponentParams] | None,
            motion_detect: MotionDetectConfig | None,
            roi: Area | None,
            fps: str | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        super().__init__(
            session=session,
            component_params=component_params,
            motion_detect=motion_detect,
            roi=roi,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            on_ready=on_ready,
            callback=callback,
            version=version
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
        if self._roi is not None:
            roi_part = mp_writer.append_json(self._roi.model_dump(exclude_none=True))
            roi_part.set_content_disposition('form-data', name='roi', filename='blob')
        if self._fps is not None:
            fps_part = mp_writer.append_json(self._fps)
            fps_part.set_content_disposition('form-data', name='fps', filename='blob')

        file_part = mp_writer.append(self.open_stream(), {'Content-Type': self.mime_type})
        file_part.set_content_disposition('form-data', name='file', filename='blob')
        return mp_writer


    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        query_params: dict[str, Any] = {
            "mode": "queue",
        }
        if self.video_mode is not None:
            query_params['videoMode'] = self.video_mode.value
        if self._version is not None:
            query_params['version'] = self._version
        if self._motion_detect is not None:
            query_params.update(self._motion_detect.model_dump(exclude_none=True))
        if self._media_cache_seconds is not None:
            query_params['mediaCacheSeconds'] = self._media_cache_seconds

        if self.needs_full_duplex:
            self._response = await session.pipeline_post(
                'prepareSource?timeout=600s',
                accept='application/jsonl',
                timeout=aiohttp.ClientTimeout(total=None, sock_read=600)
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
            query_params['processing'] = 'async'
            query_params['sourceId'] = source_id
            upload_url = f'source?{urlencode(query_params)}'
            if self._component_params is None and self._roi is None and self._fps is None:
                upload_coro = session.pipeline_post(upload_url,
                                                    accept='application/jsonl',
                                                    open_data=self.open_stream,
                                                    content_type=self.mime_type,
                                                    timeout=aiohttp.ClientTimeout(total=None, sock_read=600))
            else:
                upload_coro = session.pipeline_post(upload_url,
                                                    accept='application/jsonl',
                                                    open_data=self.open_mp_writer,
                                                    timeout=aiohttp.ClientTimeout(total=None, sock_read=600))
            read_coro = self._do_read_response(queue)
            _, got_result = await asyncio.gather(upload_coro, read_coro)
        else:
            query_params['processing'] = 'sync'
            upload_url = f'source?{urlencode(query_params)}'
            if self._component_params is None and self._roi is None and self._fps is None:
                self._response = await session.pipeline_post(upload_url,
                                                             accept='application/jsonl',
                                                             open_data=self.open_stream,
                                                             content_type=self.mime_type,
                                                             timeout=aiohttp.ClientTimeout(total=None, sock_read=600))
            else:
                self._response = await session.pipeline_post(upload_url,
                                                             accept='application/jsonl',
                                                             open_data=self.open_mp_writer,
                                                             timeout=aiohttp.ClientTimeout(total=None, sock_read=600))
            await self._do_read_response(queue)

def _guess_mime_type_from_location(location: str):
    mime_types = mimetypes.guess_type(location)
    if len(mime_types) > 0 and mime_types[0] is not None:
        mime_type = mime_types[0]
    else:
        mime_type = 'application/octet-stream'
    return mime_type


class _UploadFileJob(_UploadJob):
    def __init__(
            self,
            location: str,
            video_mode: VideoMode | None,
            component_params: list[ComponentParams] | None,
            motion_detect: MotionDetectConfig | None,
            roi: Area | None,
            fps: str | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        super().__init__(
            mime_type=_guess_mime_type_from_location(location) or 'application/octet-stream',
            open_stream=self._open_file_stream,
            video_mode=video_mode,
            component_params=component_params,
            motion_detect=motion_detect,
            roi=roi,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            session=session,
            on_ready=on_ready,
            callback=callback,
            version=version
        )
        self.location = location

    def _open_file_stream(self):
        return open(self.location, 'rb')


class _UploadStreamJob(_UploadJob):
    def __init__(
            self,
            stream: BinaryIO,
            mime_type: str,
            video_mode: VideoMode | None,
            component_params: list[ComponentParams] | None,
            motion_detect: MotionDetectConfig | None,
            roi: Area | None,
            fps: str | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        super().__init__(
            mime_type=mime_type,
            open_stream=self._get_opened_stream,
            video_mode=video_mode,
            component_params=component_params,
            motion_detect=motion_detect,
            roi=roi,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            session=session,
            on_ready=on_ready,
            callback=callback,
            version=version
        )
        self.stream = stream

    def _get_opened_stream(self):
        return self.stream


class _LoadFromJob(WorkerJob):
    def __init__(
            self,
            location: str,
            component_params: list[ComponentParams] | None,
            motion_detect: MotionDetectConfig | None,
            roi: Area | None,
            fps: str | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        super().__init__(
            session=session,
            component_params=component_params,
            motion_detect=motion_detect,
            roi=roi,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            on_ready=on_ready,
            callback=callback,
            version=version
        )
        self.location = location
        self.target_url = 'source?mode=queue&processing=sync'
        self.body: dict[str, Any] = {
            "sourceType": "URL",
            "url": self.location,
            "version": self._version,
        }
        if self._motion_detect is not None:
            self.body.update(self._motion_detect.model_dump(exclude_none=True))
        if self._component_params is not None:
            self.body['params'] = TypeAdapter(list[ComponentParams]).dump_python(self._component_params)
        if self._roi is not None:
            self.body['roi'] = self._roi.model_dump(exclude_none=True)
        if self._fps is not None:
            self.body['fps'] = self._fps
        if self._media_cache_seconds is not None:
            self.body['mediaCacheSeconds'] = self._media_cache_seconds

        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=600)

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        self._response = await session.pipeline_patch(self.target_url,
                                                      accept='application/jsonl',
                                                      data=json.dumps(self.body),
                                                      content_type='application/json',
                                                      timeout=self.timeouts)
        await self._do_read_response(queue)


class _LoadFromAssetUuidJob(WorkerJob):
    def __init__(
            self,
            asset_uuid: str,
            component_params: list[ComponentParams] | None,
            motion_detect: MotionDetectConfig | None,
            roi: Area | None,
            fps: str | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        super().__init__(
            session=session,
            component_params=component_params,
            motion_detect=motion_detect,
            roi=roi,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            on_ready=on_ready,
            callback=callback,
            version=version
        )
        self.asset_uuid = asset_uuid
        self.target_url = 'source?mode=queue&processing=sync'
        self.body: dict[str, Any] = {
            "sourceType": "ASSET_UUID",
            "assetUuid": self.asset_uuid,
            "version": self._version
        }
        if self._motion_detect is not None:
            self.body.update(self._motion_detect.model_dump())
        if self._roi is not None:
            self.body['roi'] = self._roi.model_dump(exclude_none=True)
        if self._component_params is not None:
            self.body['params'] = TypeAdapter(list[ComponentParams]).dump_python(self._component_params)
        if self._media_cache_seconds is not None:
            self.body['mediaCacheSeconds'] = self._media_cache_seconds

        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=600)

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        self._response = await session.pipeline_patch(self.target_url,
                                                      accept='application/jsonl',
                                                      data=json.dumps(self.body),
                                                      content_type='application/json',
                                                      timeout=self.timeouts)
        await self._do_read_response(queue)

