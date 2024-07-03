import asyncio
import logging
import threading
import types
import typing

from eyepop.jobs import Job

log = logging.getLogger(__name__)


class SyncJob:
    def __init__(self, job: Job, event_loop):
        self.job = job
        self.event_loop = event_loop

    def predict(self) -> dict:
        prediction = _run_coro_thread_save(self.event_loop, self.job.predict())
        return prediction

    def cancel(self):
        _run_coro_thread_save(self.event_loop, self.job.cancel())


class SyncEndpoint:
    def __init__(self, enpoint: "Endpoint"):
        self._on_ready = None
        self.endpoint = enpoint
        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, args=(self.event_loop,), daemon=True)
        self.thread.start()

    def __del__(self):
        self.event_loop.close()

    def __enter__(self) -> "SyncEndpoint":
        self.connect()
        return self

    def __exit__(
            self,
            exc_type: typing.Optional[typing.Type[BaseException]],
            exc_val: typing.Optional[BaseException],
            exc_tb: typing.Optional[types.TracebackType],
    ) -> None:
        self.disconnect()

    def _run_event_loop(self, event_loop):
        log.debug("_run_event_loop start")
        asyncio.set_event_loop(event_loop)
        event_loop.run_forever()
        log.debug("_run_event_loop done")

    def upload(self, file_path: str, params: dict | None = None,
               on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = _run_coro_thread_save(self.event_loop, self.endpoint.upload(file_path, params, None))
        return SyncJob(job, self.event_loop)

    def upload_stream(self, stream: typing.BinaryIO, mime_type: str, params: dict | None = None,
                      on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = _run_coro_thread_save(self.event_loop, self.endpoint.upload_stream(stream, mime_type, params, None))
        return SyncJob(job, self.event_loop)

    def load_from(self, location: str, params: dict | None = None,
                  on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = _run_coro_thread_save(self.event_loop, self.endpoint.load_from(location, params, None))
        return SyncJob(job, self.event_loop)

    def connect(self):
        _run_coro_thread_save(self.event_loop, self.endpoint.connect())

    def disconnect(self):
        _run_coro_thread_save(self.event_loop, self.endpoint.disconnect())

    def session(self) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.session())

    def get_pop_comp(self) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.get_pop_comp())

    def set_pop_comp(self, popComp: str) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.set_pop_comp(popComp))

    def get_post_transform(self) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.get_post_transform())

    def set_post_transform(self, transform: str) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.set_post_transform(transform))

    '''
    Start Block
    Below methods are not meant for the end user to use directly. They are used by the SDK internally.
    '''

    def list_models(self) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.list_models())

    def get_manifest(self) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.get_manifest())

    def set_manifest(self, manifest: dict) -> None:
        return _run_coro_thread_save(self.event_loop, self.endpoint.set_manifest(manifest))

    def load_model(self, model: dict, override: bool = False) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.load_model(model, override))

    def unload_model(self, model_id: str) -> None:
        return _run_coro_thread_save(self.event_loop, self.endpoint.unload_model(model_id))

    '''
    End Block
    '''


def _run_coro_thread_save(event_loop, coro):
    try:
        result = asyncio.run_coroutine_threadsafe(coro, event_loop).result()
        coro = None
        return result
    finally:
        if coro is not None:
            coro.close()
