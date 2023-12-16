import asyncio
import types
import typing

from eyepop.endpoint import Endpoint
from eyepop.jobs import Job


def syncify_endpoint(endpoint):
    return __SyncEndpoint(endpoint)


class __SyncJob:
    def __init__(self, job: Job, event_loop):
        self.job = job
        self.event_loop = event_loop

    def predict(self):
        prediction = self.event_loop.run_until_complete(self.job.predict())
        return prediction


def syncify_job(job, event_loop):
    return __SyncJob(job, event_loop)


class __SyncEndpoint:
    def __init__(self, enpoint: Endpoint):
        self.endpoint = enpoint

    def __enter__(self) -> "__SyncEndpoint":
        self.event_loop = asyncio.new_event_loop()
        self.event_loop.run_until_complete(self.endpoint.connect())
        return self

    def __exit__(
            self,
            exc_type: typing.Optional[typing.Type[BaseException]],
            exc_val: typing.Optional[BaseException],
            exc_tb: typing.Optional[types.TracebackType],
    ) -> None:
        self.event_loop.run_until_complete(self.endpoint.disconnect())
        self.event_loop.close()

    def upload(self, file_path: str) -> Job:
        job = self.event_loop.run_until_complete(self.endpoint.upload(file_path))
        return syncify_job(job, self.event_loop)

    def load_from(self, url: str) -> Job:
        job = self.event_loop.run_until_complete(self.endpoint.load_from(url))
        return syncify_job(job, self.event_loop)
