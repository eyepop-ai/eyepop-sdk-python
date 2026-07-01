import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from typing import Any, AsyncIterable, BinaryIO, Callable, cast
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
        super().__init__(session, cast(Callable[[Job], Any] | None, on_ready), callback)
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


class _UploadSource:
    """One media item to upload: a fresh-stream opener and its optional mime type.

    mime_type may be None when it cannot be derived (a raw stream group with no
    caller-supplied mime); the server no longer requires a per-member content type.
    """
    open_stream: Callable[[], BinaryIO | AsyncIterable[bytes]]
    mime_type: str | None

    def __init__(self, open_stream: Callable[[], BinaryIO | AsyncIterable[bytes]], mime_type: str | None):
        self.open_stream = open_stream
        self.mime_type = mime_type


class _UploadJob(WorkerJob):
    """Uploads one or more media items as a single source.

    A single item is posted as before: a raw body (or multipart when
    params/roi/fps are present, or full-duplex for video). Two or more items are
    posted together as one multipart request with an ordered 'file' part each -
    an image group (server SOURCE_GROUP): one inference unit, one prediction
    stream, member order preserved.
    """
    sources: list[_UploadSource]
    video_mode: VideoMode | None
    is_live: bool | None
    captured_at_offset_ns: int | None
    needs_full_duplex: bool

    def __init__(
            self,
            sources: list[_UploadSource],
            video_mode: VideoMode | None,
            is_live: bool | None,
            captured_at_offset_ns: int | None,
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
        if not sources:
            raise ValueError("upload requires at least one source")
        self.sources = sources
        self.video_mode = video_mode
        self.is_live = is_live
        self.captured_at_offset_ns = captured_at_offset_ns
        # Full duplex is only used for a single video upload; an image group is
        # always posted as one sync multipart request.
        self.needs_full_duplex = (
            len(sources) == 1
            and sources[0].mime_type is not None
            and sources[0].mime_type.startswith("video/")
        )

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

        for source in self.sources:
            if source.mime_type is not None:
                file_part = mp_writer.append(source.open_stream(), {'Content-Type': source.mime_type})
            else:
                file_part = mp_writer.append(source.open_stream())
            file_part.set_content_disposition('form-data', name='file', filename='blob')
        return mp_writer


    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
        query_params: dict[str, Any] = {
            "mode": "queue",
        }
        if self.video_mode is not None:
            query_params['videoMode'] = self.video_mode.value
        if self.is_live is not None:
            query_params["isLive"] = self.is_live
        if self.captured_at_offset_ns is not None:
            query_params['capturedAtOffsetNs'] = self.captured_at_offset_ns
        if self._version is not None:
            query_params['version'] = self._version
        if self._motion_detect is not None:
            query_params.update(self._motion_detect.model_dump(exclude_none=True))
        if self._media_cache_seconds is not None:
            query_params['mediaCacheSeconds'] = self._media_cache_seconds

        # A single item with no extra parts can stream its raw body; anything
        # else (extra parts, or a multi-item image group) is sent as multipart.
        single_source = self.sources[0] if len(self.sources) == 1 else None
        no_extra_parts = self._component_params is None and self._roi is None and self._fps is None

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
            if single_source is not None and no_extra_parts:
                upload_coro = session.pipeline_post(upload_url,
                                                    accept='application/jsonl',
                                                    open_data=single_source.open_stream,
                                                    content_type=single_source.mime_type,
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
            if single_source is not None and no_extra_parts:
                self._response = await session.pipeline_post(upload_url,
                                                             accept='application/jsonl',
                                                             open_data=single_source.open_stream,
                                                             content_type=single_source.mime_type,
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


def _file_stream_opener(location: str) -> Callable[[], BinaryIO]:
    def opener():
        return open(location, 'rb')
    return opener


def _stream_opener(stream: BinaryIO | AsyncIterable[bytes]) -> Callable[[], BinaryIO | AsyncIterable[bytes]]:
    def opener():
        return stream
    return opener


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
        self.location = location
        super().__init__(
            sources=[_UploadSource(
                _file_stream_opener(location),
                _guess_mime_type_from_location(location) or 'application/octet-stream')],
            video_mode=video_mode,
            is_live=None,
            captured_at_offset_ns=None,
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


class _UploadStreamJob(_UploadJob):
    def __init__(
            self,
            stream: BinaryIO | AsyncIterable[bytes],
            mime_type: str,
            video_mode: VideoMode | None,
            is_live: bool | None,
            captured_at_offset_ns: int | None,
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
        self.stream = stream
        super().__init__(
            sources=[_UploadSource(self._get_opened_stream, mime_type)],
            video_mode=video_mode,
            is_live=is_live,
            captured_at_offset_ns=captured_at_offset_ns,
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

    def _get_opened_stream(self):
        return self.stream


class _UploadFileGroupJob(_UploadJob):
    """Uploads multiple local images by path as a single image group.

    Thin builder over _UploadJob: one ordered _UploadSource per location (part
    order is the canonical image order), each with a Content-Type derived from
    its file extension, so it works against the current server without a
    per-member mime type. Group size and member content-type limits are enforced
    server-side.
    """

    def __init__(
            self,
            locations: list[str],
            component_params: list[ComponentParams] | None,
            roi: Area | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        sources = [
            _UploadSource(
                _file_stream_opener(location),
                _guess_mime_type_from_location(location) or 'application/octet-stream')
            for location in locations
        ]
        super().__init__(
            sources=sources,
            video_mode=None,
            is_live=None,
            captured_at_offset_ns=None,
            component_params=component_params,
            motion_detect=None,
            roi=roi,
            fps=None,
            media_cache_seconds=media_cache_seconds,
            session=session,
            on_ready=on_ready,
            callback=callback,
            version=version
        )


class _UploadStreamGroupJob(_UploadJob):
    """Uploads multiple in-memory streams as a single image group.

    Thin builder over _UploadJob: one ordered _UploadSource per stream. The
    optional mime_types list (parallel to streams) sets each part's Content-Type;
    when omitted, parts carry no explicit content type, which the server accepts.
    Group size limits are enforced server-side.
    """

    def __init__(
            self,
            streams: list[BinaryIO],
            mime_types: list[str] | None,
            component_params: list[ComponentParams] | None,
            roi: Area | None,
            media_cache_seconds: int | None,
            session: WorkerClientSession,
            on_ready: Callable[[WorkerJob], None] | None = None,
            callback: JobStateCallback | None = None,
            version: PredictionVersion = DEFAULT_PREDICTION_VERSION,
    ):
        sources = [_UploadSource(_stream_opener(stream), None) for stream in streams]
        super().__init__(
            sources=sources,
            video_mode=None,
            is_live=None,
            captured_at_offset_ns=None,
            component_params=component_params,
            motion_detect=None,
            roi=roi,
            fps=None,
            media_cache_seconds=media_cache_seconds,
            session=session,
            on_ready=on_ready,
            callback=callback,
            version=version
        )
        # Validate/apply mime types after super().__init__ so a bad call does not
        # leave a half-constructed Job behind.
        if mime_types is not None:
            if len(mime_types) != len(streams):
                raise ValueError("mime_types must have the same length as streams")
            for source, mime_type in zip(self.sources, mime_types, strict=True):
                source.mime_type = mime_type


class _LoadFromJob(WorkerJob):
    """Loads one or more server-fetched URLs as a single source.

    A single URL is a normal `URL` source (unchanged, including video/motion/fps
    handling). Two or more URLs form one `GROUP` source - an image group run as a
    single inference unit, member order preserved. The server fetches each URL and
    reads its own Content-Type, so no per-member mime type is sent.
    """

    def __init__(
            self,
            locations: list[str],
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
        if not locations:
            raise ValueError("load_from requires at least one url")
        self.locations = list(locations)
        self.target_url = 'source?mode=queue&processing=sync'
        if len(self.locations) == 1:
            # Single URL: unchanged body (field order preserved for compatibility).
            self.body: dict[str, Any] = {
                "sourceType": "URL",
                "url": self.locations[0],
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
        else:
            # Two or more URLs: one image group (no video/motion/fps).
            self.body = {
                "sourceType": "GROUP",
                "sources": [{"sourceType": "URL", "url": location} for location in self.locations],
                "version": self._version,
            }
            if self._component_params is not None:
                self.body['params'] = TypeAdapter(list[ComponentParams]).dump_python(self._component_params)
            if self._roi is not None:
                self.body['roi'] = self._roi.model_dump(exclude_none=True)
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
