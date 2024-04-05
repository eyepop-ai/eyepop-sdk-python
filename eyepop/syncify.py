import asyncio
import types
import typing

from eyepop.jobs import Job


class SyncJob:
    def __init__(self, job: Job, event_loop):
        self.job = job
        self.event_loop = event_loop

    def predict(self) -> dict:
        prediction = self.event_loop.run_until_complete(self.job.predict())
        return prediction

    def cancel(self):
        self.event_loop.run_until_complete(self.job.cancel())


class SyncEndpoint:
    def __init__(self, enpoint: "Endpoint"):
        self._on_ready = None
        self.endpoint = enpoint
        self.event_loop = asyncio.new_event_loop()

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

    def upload(self, file_path: str, params: dict | None = None, on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = self.event_loop.run_until_complete(self.endpoint.upload(file_path, params, None))
        return SyncJob(job, self.event_loop)

    def upload_stream(self, stream: typing.BinaryIO, mime_type: str, params: dict | None = None, on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = self.event_loop.run_until_complete(self.endpoint.upload_stream(stream, mime_type, params, None))
        return SyncJob(job, self.event_loop)

    def load_from(self, location: str, params: dict | None = None, on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = self.event_loop.run_until_complete(self.endpoint.load_from(location, params, None))
        return SyncJob(job, self.event_loop)

    def connect(self):
        self.event_loop.run_until_complete(self.endpoint.connect())

    def disconnect(self):
        self.event_loop.run_until_complete(self.endpoint.disconnect())

    def get_pop_comp(self) -> dict:
        return self.event_loop.run_until_complete(self.endpoint.get_pop_comp())
    
    def set_pop_comp(self, popComp: str) -> dict:
        return self.event_loop.run_until_complete(self.endpoint.set_pop_comp(popComp))

    '''
    Start Block
    Below methods are not meant for the end user to use directly. They are used by the SDK internally.
    '''
    def list_models(self) -> dict:
        return self.event_loop.run_until_complete(self.endpoint.list_models())
    
    def get_manifest(self) -> dict:
        return self.event_loop.run_until_complete(self.endpoint.get_manifest())
    
    def set_manifest(self, manifest:dict) -> None:
        return self.event_loop.run_until_complete(self.endpoint.set_manifest(manifest))
    
    def load_model(self, model:dict, override:bool = False) -> None:
        return self.event_loop.run_until_complete(self.endpoint.load_model(model, override))
    '''
    End Block
    '''