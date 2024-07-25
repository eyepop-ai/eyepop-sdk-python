import asyncio
from asyncio import Queue
from enum import Enum
from typing import Callable

from eyepop.client_session import ClientSession
from eyepop.worker.worker_client_session import WorkerClientSession


class JobState(Enum):
    CREATED = 1
    STARTED = 2
    IN_PROGRESS = 3
    FINISHED = 4
    FAILED = 5
    DRAINED = 6

    def __repr__(self):
        return self._name_


class JobStateCallback:
    def created(self, job):
        pass

    def started(self, job):
        pass

    def first_result(self, job):
        pass

    def failed(self, job):
        pass

    def finished(self, job):
        pass

    def drained(self, job):
        pass

    def finalized(self, job):
        pass


class Job:
    """
    Abstract Job submitted to an EyePop.ai Endpoint.
    """

    def __init__(self,
                 session: ClientSession,
                 on_ready: Callable[["Job"], None] | None,
                 callback: JobStateCallback | None = None):
        self.on_ready = on_ready
        self._session = session
        self._response = None
        self._queue = asyncio.Queue(maxsize=128)
        if callback is not None:
            self._callback = callback
        else:
            self._callback = JobStateCallback()
        self._callback.created(self)

    def __del__(self):
        self._callback.finalized(self)

    async def push_result(self, result) -> None:
        await self._queue.put(result)

    async def pop_result(self) -> dict:
        queue = self._queue
        if queue is None:
            return None
        else:
            result = await queue.get()
            if result is None:
                self._queue = None
                self._callback.drained(self)
            elif isinstance(result, Exception):
                self._queue = None
                self._callback.drained(self)
                raise result
            return result

    async def cancel(self):
        queue = self._queue
        if queue is None:
            return None
        self._queue = None
        if self._response is not None:
            self._response.close()
        await queue.put(None)

    async def execute(self):
        queue = self._queue
        session = self._session

        self._callback.started(self)

        try:
            await self._do_execute_job(queue, session)
            await queue.put(None)
        except Exception as e:
            if self._queue is None:
                # we got canceled
                pass
            else:
                self._callback.failed(self)
                await queue.put(e)
        finally:
            if self._response is not None:
                response = self._response.close()
                if response is not None:
                    response.release()
            self._callback.finished(self)
            if self.on_ready is not None:
                await self.on_ready(self)

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        raise NotImplementedError("can't execute abstract jobs")

