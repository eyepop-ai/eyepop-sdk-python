import logging
import typing

from eyepop.syncify import run_coro_thread_save, SyncEndpoint
from eyepop.worker.worker_jobs import WorkerJob

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

    def upload(self, file_path: str, params: dict | None = None,
               on_ready: typing.Callable[[WorkerJob], None] | None = None) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload(file_path, params, None))
        return SyncWorkerJob(job, self.event_loop)

    def upload_stream(self, stream: typing.BinaryIO, mime_type: str, params: dict | None = None,
                      on_ready: typing.Callable[[WorkerJob], None] | None = None) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.upload_stream(stream, mime_type, params, None))
        return SyncWorkerJob(job, self.event_loop)

    def load_from(self, location: str, params: dict | None = None,
                  on_ready: typing.Callable[[WorkerJob], None] | None = None) -> SyncWorkerJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.workerEndpoint(is_async=True)` to create an async endpoint with callback support")
        job = run_coro_thread_save(self.event_loop, self.endpoint.load_from(location, params, None))
        return SyncWorkerJob(job, self.event_loop)

    def get_pop_comp(self) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_pop_comp())

    def set_pop_comp(self, popComp: str) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.set_pop_comp(popComp))

    def get_post_transform(self) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_post_transform())

    def set_post_transform(self, transform: str) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.set_post_transform(transform))

    '''
    Start Block
    Below methods are not meant for the end user to use directly. They are used by the SDK internally.
    '''

    def list_models(self) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.list_models())

    def get_manifest(self) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_manifest())

    def set_manifest(self, manifest: dict) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.set_manifest(manifest))

    def load_model(self, model: dict, override: bool = False) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.load_model(model, override))

    def unload_model(self, model_id: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.unload_model(model_id))

    '''
    End Block
    '''


