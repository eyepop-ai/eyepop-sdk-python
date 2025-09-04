import logging
import typing

from eyepop.syncify import run_coro_thread_save, SyncEndpoint
from eyepop.worker.worker_jobs import WorkerJob
from eyepop.worker.worker_types import Pop, VideoMode, ComponentParams

if typing.TYPE_CHECKING:
    from eyepop.worker.worker_endpoint import WorkerEndpoint

log = logging.getLogger(__name__)


class SyncWorkerJob:
    def __init__(self, job: WorkerJob, event_loop):
        self.job = job
        self.event_loop = event_loop

    def predict(self) -> dict:
        prediction = run_coro_thread_save(self.event_loop, self.job.predict())
        return prediction

    def cancel(self):
        run_coro_thread_save(self.event_loop, self.job.cancel())


class SyncWorkerEndpoint(SyncEndpoint):
    def __init__(self, endpoint: "WorkerEndpoint"):
        super().__init__(endpoint)

    def upload(
            self,
            location: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload(
            location=location, video_mode=video_mode, params=params, on_ready=None))
        return SyncWorkerJob(job, self.event_loop)

    def upload_stream(
            self,
            stream: typing.BinaryIO,
            mime_type: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload_stream(
            stream=stream, mime_type=mime_type, video_mode=video_mode, params=params, on_ready=None
        ))
        return SyncWorkerJob(job, self.event_loop)

    def load_from(
            self,
            location: str,
            params: list[ComponentParams] | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.load_from(
            location=location, params=params, on_ready=None))
        return SyncWorkerJob(job, self.event_loop)

    def load_asset(
            self,
            asset_uuid: str,
            params: list[ComponentParams] | None = None,
            on_ready: typing.Callable[[WorkerJob], None] | None = None
    ) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.load_asset(
            asset_uuid=asset_uuid, params=params, on_ready=None))
        return SyncWorkerJob(job, self.event_loop)

    def get_pop(self) -> Pop | None:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_pop())

    def set_pop(self, pop: Pop) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.set_pop(pop))


