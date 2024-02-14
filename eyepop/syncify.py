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

    def upload(self, file_path: str, on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = self.event_loop.run_until_complete(self.endpoint.upload(file_path, None))
        return SyncJob(job, self.event_loop)

    def upload_stream(self, stream: typing.BinaryIO, mime_type: str, on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = self.event_loop.run_until_complete(self.endpoint.upload_stream(stream, mime_type, None))
        return SyncJob(job, self.event_loop)

    def load_from(self, location: str, on_ready: typing.Callable[[Job], None] | None = None) -> SyncJob:
        if on_ready is not None:
            raise TypeError(
                "'on_ready' callback not supported for sync endpoints. "
                "Use 'EyePopSdk.connect(is_async=True)` to create an async endpoint with callback support")
        job = self.event_loop.run_until_complete(self.endpoint.load_from(location, None))
        return SyncJob(job, self.event_loop)

    def connect(self):
        self.event_loop.run_until_complete(self.endpoint.connect())

    def disconnect(self):
        self.event_loop.run_until_complete(self.endpoint.disconnect())
