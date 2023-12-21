import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from enum import Enum
from typing import Callable

import aiohttp
from aiohttp import ClientSession

log = logging.getLogger('eyepop')


class JobType(Enum):
    UPLOAD = 1
    FROM_URL = 2


class Job:
    """
    Abstract Job submitted to an EyePop.ai Endpoint.
    """

    def __init__(self, job_type: JobType, location: str, session: ClientSession,
                 on_ready: Callable[["Job"], None] | None):
        self.on_ready = on_ready
        self.job_type = job_type
        self.location = location
        self._session = session
        self._response = None
        self._queue = asyncio.Queue(maxsize=128)

    async def predict(self) -> dict:
        queue = self._queue
        if queue is None:
            return None
        else:
            prediction = await queue.get()
            if prediction is None:
                self._queue = None
            elif isinstance(prediction, Exception):
                self._queue = None
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

        try:
            await self._do_execute_job(queue, session)
            await queue.put(None)
        except Exception as e:
            if self._queue is None:
                # we got canceled
                pass
            else:
                await queue.put(e)
        finally:
            if self._response is not None:
                response = self._response.close()
                if response is not None:
                    response.release()
            if self.on_ready:
                await self.on_ready(self)

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        raise NotImplementedError("can't execute abstract jobs")


class _UploadJob(Job):
    def __init__(self, location: str, pipeline_base_url: str, authorization_header: str, session: ClientSession,
                 on_ready: Callable[[Job], None] | None = None):
        super().__init__(JobType.UPLOAD, location, session, on_ready)
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
            self._response = await session.post(self._target_url, headers=self._headers, data=file,
                                                timeout=self.timeouts)
            while line := await self._response.content.readline():
                prediction = json.loads(line)
                await queue.put(prediction)


class _LoadFromJob(Job):
    def __init__(self, location: str, pipeline_base_url: str, authorization_header: str, session: ClientSession,
                 on_ready: Callable[[Job], None] | None = None):
        super().__init__(JobType.FROM_URL, location, session, on_ready)
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
        self._response = await session.patch(self._target_url, headers=self._headers, json=self._body,
                                             timeout=self.timeouts)
        while line := await self._response.content.readline():
            prediction = json.loads(line)
            await queue.put(prediction)
