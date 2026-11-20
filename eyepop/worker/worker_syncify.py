import logging
import typing
from collections.abc import Iterator, Mapping, Sequence

from eyepop.data.types.asset import Area
from eyepop.syncify import SyncEndpoint, run_coro_thread_save
from eyepop.worker.camera import Camera
from eyepop.worker.media import ClipMode, MediaClip
from eyepop.worker.worker_jobs import WorkerJob
from eyepop.worker.worker_types import ComponentParams, MotionDetectConfig, Pop, VideoMode

if typing.TYPE_CHECKING:
    from eyepop.worker.worker_endpoint import WorkerEndpoint

log = logging.getLogger(__name__)


class SyncWorkerJob:
    def __init__(self, job: WorkerJob, event_loop):
        self.job = job
        self.event_loop = event_loop

    def predict(self) -> dict | None:
        prediction = run_coro_thread_save(self.event_loop, self.job.predict())
        return prediction

    def cancel(self):
        run_coro_thread_save(self.event_loop, self.job.cancel())

    @property
    def media_id(self) -> str | None:
        """The recording's mediaId, once the source started; see WorkerJob.media_id."""
        return self.job.media_id

    @property
    def warnings(self) -> list[dict]:
        """The warnings the pipeline sent about the source; see WorkerJob.warnings."""
        return self.job.warnings

    def media_frame(
            self,
            prediction: Mapping[str, typing.Any],
            obj: Mapping[str, typing.Any] | None = None,
            *,
            max_width: int | None = None,
            max_height: int | None = None,
            png: bool = False,
            wait_ms: int | None = None,
    ) -> bytes:
        """The recorded frame a prediction was made on, cropped to `obj`; see WorkerJob.media_frame."""
        frame: bytes = run_coro_thread_save(self.event_loop, self.job.media_frame(
            prediction, obj, max_width=max_width, max_height=max_height, png=png, wait_ms=wait_ms))
        return frame


class SyncMediaClip:
    """A clip of a recording, read in the sync endpoint's thread; see MediaClip.

    Iterate it for its bytes, `read()` it whole, or use it as a context manager, which closes it.
    """

    def __init__(self, clip: MediaClip, event_loop):
        self.clip = clip
        self.event_loop = event_loop

    @property
    def start_ns(self) -> int | None:
        return self.clip.start_ns

    @property
    def end_ns(self) -> int | None:
        return self.clip.end_ns

    @property
    def milli_compute_units(self) -> int | None:
        return self.clip.milli_compute_units

    @property
    def content_type(self) -> str:
        return self.clip.content_type

    def read(self) -> bytes:
        data: bytes = run_coro_thread_save(self.event_loop, self.clip.read())
        return data

    def __iter__(self) -> Iterator[bytes]:
        """The clip's bytes, chunk by chunk."""
        chunks = aiter(self.clip)

        async def next_chunk() -> bytes | None:
            try:
                return await anext(chunks)
            except StopAsyncIteration:
                return None

        while (chunk := run_coro_thread_save(self.event_loop, next_chunk())) is not None:
            yield chunk

    def close(self) -> None:
        async def close():
            # in the endpoint's loop, which owns the connection
            self.clip.close()
        run_coro_thread_save(self.event_loop, close())

    def __enter__(self) -> "SyncMediaClip":
        """The clip, closed when the block ends."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the clip."""
        self.close()


class SyncWorkerEndpoint(SyncEndpoint):
    def __init__(self, endpoint: "WorkerEndpoint"):
        super().__init__(endpoint)

    def upload(
            self,
            location: str,
            video_mode: VideoMode | None = None,
            is_live: bool | None = None,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            *,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload(
            location=location,
            video_mode=video_mode,
            is_live=is_live,
            params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def upload_stream(
            self,
            stream: typing.BinaryIO,
            mime_type: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            # Keyword-only: this method published its positional order without
            # them, and `params` must keep its slot.
            *,
            is_live: bool | None = None,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload_stream(
            stream=stream,
            mime_type=mime_type,
            video_mode=video_mode,
            is_live=is_live,
            params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def upload_stream_group(
            self,
            streams: list[typing.BinaryIO],
            mime_types: list[str] | None = None,
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            *,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload_stream_group(
            streams=streams,
            mime_types=mime_types,
            params=params,
            roi=roi,
            camera=camera,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def upload_group(
            self,
            locations: list[str],
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            *,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload_group(
            locations=locations,
            params=params,
            roi=roi,
            camera=camera,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def load_from(
            self,
            location: str,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            rtsp_force_non_compliant_url: bool | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            *,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.load_from(
            location=location,
            params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            rtsp_force_non_compliant_url=rtsp_force_non_compliant_url,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def load_from_group(
            self,
            locations: list[str],
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            *,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.load_from_group(
            locations=locations,
            params=params,
            roi=roi,
            camera=camera,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def load_asset(
            self,
            asset_uuid: str,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None,
            *,
            media_cache_reencode: bool | None = None,
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.async_worker()` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.load_asset(
            asset_uuid=asset_uuid,
            params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            media_cache_reencode=media_cache_reencode,
            on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def media_range(self, media_id: str) -> tuple[int, int]:
        """The recording's first and last nanosecond; see WorkerEndpoint.media_range."""
        bounds: tuple[int, int] = run_coro_thread_save(self.event_loop, self.endpoint.media_range(media_id))
        return bounds

    def media_frame(
            self,
            media_id: str,
            timestamp_ns: int,
            *,
            bbox: Sequence[float] | None = None,
            max_width: int | None = None,
            max_height: int | None = None,
            png: bool = False,
            wait_ms: int | None = None,
    ) -> bytes:
        """The recorded frame at a prediction's timestamp; see WorkerEndpoint.media_frame."""
        frame: bytes = run_coro_thread_save(self.event_loop, self.endpoint.media_frame(
            media_id, timestamp_ns, bbox=bbox, max_width=max_width, max_height=max_height, png=png,
            wait_ms=wait_ms))
        return frame

    def media_clip(
            self,
            media_id: str,
            start_ns: int,
            end_ns: int,
            *,
            mode: ClipMode = 'exact',
            wait_ms: int | None = None,
    ) -> SyncMediaClip:
        """An MP4 clip of a recording; see WorkerEndpoint.media_clip."""
        clip = run_coro_thread_save(self.event_loop, self.endpoint.media_clip(
            media_id, start_ns, end_ns, mode=mode, wait_ms=wait_ms))
        return SyncMediaClip(clip, self.event_loop)

    def get_pop(self) -> Pop | None:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_pop())

    def set_pop(self, pop: Pop) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.set_pop(pop))

