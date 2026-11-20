import logging
import mimetypes
from asyncio import Queue
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterable, BinaryIO, Callable, cast

import aiohttp

from eyepop.client_session import ClientSession
from eyepop.data.types.asset import Area
from eyepop.jobs import Job, JobStateCallback
from eyepop.worker.camera import Camera
from eyepop.worker.media import frame_box, read_frame
from eyepop.worker.worker_types import (
    DEFAULT_PREDICTION_VERSION,
    ComponentParams,
    MotionDetectConfig,
    PredictionVersion,
    VideoMode,
)

if TYPE_CHECKING:
    from eyepop.worker.worker_backend import WorkerBackend

log = logging.getLogger('eyepop')
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
    media_cache_reencode: bool | None = None
    rtsp_force_non_compliant_url: bool | None = None
    version: PredictionVersion = DEFAULT_PREDICTION_VERSION

    def live(self) -> bool:
        """Whether the source runs until it is stopped rather than to an end of its own."""
        return bool(self.is_live) or any(url.startswith(('rtsp://', 'rtmp://')) for url in self.urls)


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
        self.milli_compute_units: int | None = None
        """What the source's remote calls (a VLM ability's calls to vlm-server) used so far, in milli compute
        units, in pipeline mode: the running total while the source runs, its total once it ended. None until the
        pipeline API reports any; a local model reports none. Informational: the called API bills its calls."""
        self.media_id: str | None = None
        """The recording's mediaId, once a source that records to the media cache (`media_cache_seconds`)
        has started: an opaque string to pass back unchanged to read the recording, with `media_frame`
        here or the endpoint's `media_range`, `media_frame` and `media_clip`. In pipeline mode it comes with
        the source's `start` (`v1.`), on a session with the `recording` event that opens its result stream
        (`v2.`, from an instance with eyepop-instance spec 025). None until then, and for a source that
        does not record."""
        self.warnings: list[dict[str, Any]] = []
        """The warnings the pipeline sent about the source, in order, in both modes, each with its
        `message` and, when it has one, its `info` (a `code`, a `level` and details, such as `gap_ms` for
        `media_cache_gop_too_long`). The source still runs; each is also logged once on the `eyepop`
        logger."""

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
                if event is None:
                    return result
                # An event is the pipeline's line, {"event": "<type>", ...}, or the
                # instance's own, {"event": {"type": "<type>", ...}}. Only an error ends the
                # stream; any other event, including a type added after this SDK, is skipped.
                # A session's recording and warning events were kept on the job as they
                # arrived (note_event).
                details, type_ = _event_of(event, result)
                if type_ == 'error':
                    source_id = details.get('source_id', None)
                    message = details.get('message', None)
                    raise ValueError(f"Error in source {source_id}: {message}")

    def note_event(self, line: Any) -> None:
        """Keep what a session's event line tells the job: the recording's mediaId and the pipeline's warnings.

        A session's result stream carries the instance's `{"event": {"type": "recording", "mediaId": ...}}`
        as its first line for a source that records (spec 025), and the pipeline's
        `{"event": "warning", "message": ..., "info": ...}` as it comes. Pipeline mode keeps both from the
        socket's messages instead.
        """
        if not isinstance(line, dict) or line.get('event') is None:
            return
        details, type_ = _event_of(line['event'], line)
        if type_ == 'recording':
            media_id = details.get('mediaId')
            if isinstance(media_id, str):
                self.media_id = media_id
        elif type_ == 'warning':
            warning: dict[str, Any] = {'message': details.get('message', '')}
            if isinstance(details.get('info'), dict):
                warning['info'] = details['info']
            self.warnings.append(warning)
            log.warning('the pipeline warns about source %s: %s', details.get('source_id'), warning['message'])

    async def media_frame(
            self,
            prediction: Mapping[str, Any],
            obj: Mapping[str, Any] | None = None,
            *,
            max_width: int | None = None,
            max_height: int | None = None,
            png: bool = False,
            wait_ms: int | None = None,
    ) -> bytes:
        """The recorded frame a prediction was made on, cropped to one of its objects when given.

        Reads the job's recording (`media_id`) at the prediction's `timestamp` (nanoseconds), as the
        endpoint's `media_frame` does. `obj` is an object of the prediction, at any depth; its `x`, `y`,
        `width` and `height` are in the source's pixels, which are the recording's frame pixels, and the
        crop is the smallest box of whole pixels that holds them. The other arguments are `media_frame`'s.
        Raises ValueError when the job has no `media_id` (its source does not record, or has not started)
        or the prediction has no timestamp, and MediaReadException when the read is refused.
        """
        if self.media_id is None:
            raise ValueError('the job has no media_id: its source does not record to the media cache '
                             '(media_cache_seconds) or has not started yet')
        timestamp = prediction.get('timestamp')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise ValueError(f'the prediction has no timestamp in nanoseconds: {timestamp!r}')
        return await read_frame(
            self.backend, self.media_id, int(timestamp),
            bbox=frame_box(obj) if obj is not None else None,
            max_width=max_width, max_height=max_height, png=png, wait_ms=wait_ms)

    def set_response(self, response: Any):
        """The response a backend reads predictions from; cancel() closes it."""
        self._response = response

    async def cancel(self):
        await super().cancel()
        await self.backend.cancelled(self)

    async def _do_execute_job(self, queue: Queue, session: Any):
        await self.backend.run(self)


def _event_of(event: Any, line: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """An event line's details and type: the instance's {"event": {"type": ...}} or the pipeline's {"event": "<type>"}."""
    if isinstance(event, dict):
        return event, event.get('type', None)
    return line, event


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
