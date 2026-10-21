import logging
import mimetypes
from asyncio import Queue
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterable, BinaryIO, Callable, cast

import aiohttp

from eyepop.client_session import ClientSession
from eyepop.data.types.asset import Area
from eyepop.jobs import Job, JobStateCallback
from eyepop.worker.camera import Camera
from eyepop.worker.worker_types import (
    DEFAULT_PREDICTION_VERSION,
    ComponentParams,
    MotionDetectConfig,
    PredictionVersion,
    VideoMode,
)

if TYPE_CHECKING:
    from eyepop.worker.worker_backend import WorkerBackend

log_requests = logging.getLogger('eyepop.requests')


async def _iter_lines(content: aiohttp.StreamReader) -> AsyncIterable[bytes]:
    """Yield the newline delimited lines of a response body.

    aiohttp's `readline()` raises ValueError('Chunk too big') for lines longer
    than the session's read_bufsize (64kb by default). Predictions can carry
    large binary members - a single depth map is ~1mb of base64 per frame - so
    lines are accumulated here instead, without a size limit.
    """
    buffer = bytearray()
    async for chunk in content.iter_any():
        buffer.extend(chunk)
        start = 0
        while (index := buffer.find(b'\n', start)) >= 0:
            yield bytes(buffer[start:index + 1])
            start = index + 1
        del buffer[:start]
    if buffer:
        yield bytes(buffer)


class _UploadSource:
    """One media item to upload: a fresh-stream opener, its optional mime type and, for a file, its path.

    mime_type may be None when it cannot be derived (a raw stream group with no
    caller-supplied mime); the server no longer requires a per-member content type.
    """
    open_stream: Callable[[], BinaryIO | AsyncIterable[bytes]]
    mime_type: str | None
    location: str | None

    def __init__(self, open_stream: Callable[[], BinaryIO | AsyncIterable[bytes]], mime_type: str | None,
                 location: str | None = None):
        self.open_stream = open_stream
        self.mime_type = mime_type
        self.location = location


@dataclass
class WorkerSource:
    """What a job runs, whatever backend runs it.

    The media is one of `urls` (server-fetched), `asset_uuid` or `uploads`;
    `group` says it came from a group method, whose items run as one inference
    unit.
    """
    urls: list[str] = field(default_factory=list)
    asset_uuid: str | None = None
    uploads: list[_UploadSource] = field(default_factory=list)
    group: bool = False
    video_mode: VideoMode | None = None
    is_live: bool | None = None
    component_params: list[ComponentParams] | None = None
    motion_detect: MotionDetectConfig | None = None
    roi: Area | None = None
    fps: str | None = None
    camera: Camera | None = None
    media_cache_seconds: int | None = None
    rtsp_force_non_compliant_url: bool | None = None
    version: PredictionVersion = DEFAULT_PREDICTION_VERSION


class WorkerJob(Job):
    """A job submitted to an EyePop.ai WorkerEndpoint, run by the endpoint's backend."""

    def __init__(
            self,
            session: ClientSession,
            backend: 'WorkerBackend',
            source: WorkerSource,
            on_ready: Callable[["WorkerJob"], None] | None,
            callback: JobStateCallback | None = None,
    ):
        super().__init__(session, cast(Callable[[Job], Any] | None, on_ready), callback)
        self.backend = backend
        self.source = source

    @property
    def cancelled(self) -> bool:
        return self._queue is None

    @property
    def callback(self) -> JobStateCallback:
        return self._callback

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

    def set_response(self, response: Any):
        """The response a backend reads predictions from; cancel() closes it."""
        self._response = response

    async def cancel(self):
        await super().cancel()
        await self.backend.cancelled(self)

    async def _do_execute_job(self, queue: Queue, session: Any):
        await self.backend.run(self)


def _guess_mime_type_from_location(location: str):
    mime_types = mimetypes.guess_type(location)
    if len(mime_types) > 0 and mime_types[0] is not None:
        mime_type = mime_types[0]
    else:
        mime_type = 'application/octet-stream'
    return mime_type


def _file_upload(location: str) -> _UploadSource:
    def opener():
        return open(location, 'rb')
    return _UploadSource(opener, _guess_mime_type_from_location(location) or 'application/octet-stream', location)


def _stream_upload(stream: BinaryIO | AsyncIterable[bytes], mime_type: str | None) -> _UploadSource:
    def opener():
        return stream
    return _UploadSource(opener, mime_type)
