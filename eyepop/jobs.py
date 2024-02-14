import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from enum import Enum
from typing import Callable, BinaryIO

import aiohttp
from aiohttp import ClientSession

log_requests = logging.getLogger('eyepop.requests')


class _JobStateCallback:
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
                 callback: _JobStateCallback | None = None):
        self.on_ready = on_ready
        self._session = session
        self._response = None
        self._queue = asyncio.Queue(maxsize=128)
        if callback is not None:
            self._callback = callback
        else:
            self._callback = _JobStateCallback()
        self._callback.created(self)

    def __del__(self):
        self._callback.finalized(self)

    async def predict(self) -> dict:
        queue = self._queue
        if queue is None:
            return None
        else:
            prediction = await queue.get()
            if prediction is None:
                self._queue = None
                self._callback.drained(self)
            elif isinstance(prediction, Exception):
                self._queue = None
                self._callback.drained(self)
                raise prediction
            return prediction

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
            if self.on_ready:
                await self.on_ready(self)

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        raise NotImplementedError("can't execute abstract jobs")

    async def _do_read_response(self, queue: Queue):
        got_results = False
        async with self._response as response:
            self._callback.first_result(self)
            while line := await response.content.readline():
                if not got_results:
                    got_results = True
                prediction = json.loads(line)
                await queue.put(prediction)


class _UploadJob(Job):
    def __init__(self, location: str, pipeline_base_url: str, authorization_header: str, session: ClientSession,
                 on_ready: Callable[[Job], None] | None = None, callback: _JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self.location = location
        mime_types = mimetypes.guess_type(location)
        if len(mime_types) > 0:
            mime_type = mime_types[0]
        else:
            mime_type = 'application/octet-stream'
        self._target_url = f'{pipeline_base_url}/source?mode=queue&processing=sync'
        self._headers = {
            'Content-Type': mime_type,
            'Accept': 'application/jsonl',
            'Authorization': authorization_header
        }
        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=60)

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        with open(self.location, 'rb') as file:
            log_requests.debug("before POST %s with file %s as body", self._target_url, self.location)
            self._response = await session.post(self._target_url, headers=self._headers, data=file,
                                                timeout=self.timeouts)
            await self._do_read_response(queue)
            log_requests.debug("after POST %s with file %s as body", self._target_url, self.location)

class _UploadStreamJob(Job):
    def __init__(self, stream: BinaryIO, mime_type: str, pipeline_base_url: str, authorization_header: str, session: ClientSession,
                 on_ready: Callable[[Job], None] | None = None, callback: _JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self.stream = stream
        self.mime_type = mime_type
        self._target_url = f'{pipeline_base_url}/source?mode=queue&processing=sync'
        self._headers = {
            'Content-Type': self.mime_type,
            'Accept': 'application/jsonl',
            'Authorization': authorization_header
        }
        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=60)

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        log_requests.debug("before POST %s with stream as body with mime type %s", self._target_url, self.mime_type)
        self._response = await session.post(self._target_url, headers=self._headers, data=self.stream,
                                            timeout=self.timeouts)
        await self._do_read_response(queue)
        log_requests.debug("after POST %s with stream as body with mime type %s", self._target_url, self.mime_type)

class _LoadFromJob(Job):
    def __init__(self, location: str, pipeline_base_url: str, authorization_header: str, session: ClientSession,
                 on_ready: Callable[[Job], None] | None = None, callback: _JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self.location = location
        self._target_url = f'{pipeline_base_url}/source?mode=queue&processing=sync'
        self._headers = {
            'Accept': 'application/jsonl',
            'Authorization': authorization_header
        }
        self._body = {
            "sourceType": "URL",
            "url": self.location
        }
        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=60)

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        log_requests.debug("before PATCH %s with url %s as source", self._target_url, self.location)
        self._response = await session.patch(self._target_url, headers=self._headers, json=self._body,
                                             timeout=self.timeouts)
        await self._do_read_response(queue)
        log_requests.debug("after PATCH %s with url %s as source", self._target_url, self.location)
