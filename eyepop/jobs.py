import asyncio
import json
import logging
import mimetypes
from asyncio import Queue
from enum import Enum
from typing import Callable, BinaryIO, Any

import aiohttp

log_requests = logging.getLogger('eyepop.requests')


class _WorkerClientSession:
    async def pipeline_get(
            self, url_path_and_query: str,
            accept: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass

    async def pipeline_post(
            self, url_path_and_query: str,
            accept: str | None = None,
            open_data: Callable = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass

    async def pipeline_patch(
            self, url_path_and_query: str,
            accept: str | None = None,
            data: Any = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass

    async def pipeline_delete(
            self, url_path_and_query: str,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass


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
                 session: _WorkerClientSession,
                 params: dict | None,
                 on_ready: Callable[["Job"], None] | None,
                 callback: JobStateCallback | None = None):
        self.on_ready = on_ready
        self._session = session
        self._params = params
        self._response = None
        self._queue = asyncio.Queue(maxsize=128)
        if callback is not None:
            self._callback = callback
        else:
            self._callback = JobStateCallback()
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
            if self.on_ready is not None:
                await self.on_ready(self)

    async def _do_execute_job(self, queue: Queue, session: _WorkerClientSession):
        raise NotImplementedError("can't execute abstract jobs")

    async def _do_read_response(self, queue: Queue) -> bool:
        got_results = False
        if self._response is not None:
            async with self._response as response:
                self._callback.first_result(self)
                while line := await response.content.readline():

                    # TODO aiohttp should do do this internally
                    for trace in response._traces:
                        await trace.send_response_chunk_received(
                            response.method, response.url, line
                        )

                    if not got_results:
                        got_results = True
                    prediction = json.loads(line)
                    await queue.put(prediction)
        return got_results


class _UploadJob(Job):
    def __init__(self, location: str, params: dict | None,
                 session: _WorkerClientSession,
                 on_ready: Callable[[Job], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.location = location
        self.target_url = 'source?mode=queue&processing=sync'
        mime_types = mimetypes.guess_type(location)
        if len(mime_types) > 0:
            self.mime_type = mime_types[0]
        else:
            self.mime_type = 'application/octet-stream'

    async def _do_execute_job(self, queue: Queue, session: _WorkerClientSession):
        def open_file():
            return open(self.location, 'rb')

        def open_mp_writer():
            mp_writer = aiohttp.MultipartWriter('form-data')
            params_part = mp_writer.append_json(self._params)
            params_part.set_content_disposition('form-data', name='params', filename='blob')
            file_part = mp_writer.append(open_file(), {'Content-Type': self.mime_type})
            file_part.set_content_disposition('form-data', name='file', filename='blob')

        try:
            got_result = False
            if self._params is None:
                self._response = await session.pipeline_post(self.target_url,
                                                             accept='application/jsonl',
                                                             open_data=open_file,
                                                             content_type=self.mime_type,
                                                             timeout=aiohttp.ClientTimeout(total=None, sock_read=60))
            else:
                self._response = await session.pipeline_post(self.target_url,
                                                             accept='application/jsonl',
                                                             open_data=open_mp_writer,
                                                             timeout=aiohttp.ClientTimeout(total=None, sock_read=60))

            got_result = await self._do_read_response(queue)
        finally:
            if not got_result:
                pass
                # await queue.put(None)


class _UploadStreamJob(Job):
    def __init__(self, stream: BinaryIO, mime_type: str, params: dict | None,
                 session: _WorkerClientSession,
                 on_ready: Callable[[Job], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.stream = stream
        self.mime_type = mime_type
        self.target_url = 'source?mode=queue&processing=sync'

    async def _do_execute_job(self, queue: Queue, session: _WorkerClientSession):
        if self._params is None:
            self._response = await session.pipeline_post(self.target_url,
                                                         accept='application/jsonl',
                                                         content_type=self.mime_type,
                                                         open_data=lambda: self.stream,
                                                         timeout=aiohttp.ClientTimeout(total=None, sock_read=60))
        else:
            with aiohttp.MultipartWriter('form-data') as mp_writer:
                params_part = mp_writer.append_json(self._params)
                params_part.set_content_disposition('form-data', name='params')
                file_part = mp_writer.append(self.stream, {'Content-Type': self.mime_type})
                file_part.set_content_disposition('form-data', name='file')
                self._response = await session.pipeline_post(self.target_url,
                                                             accept='application/jsonl',
                                                             open_data=lambda: mp_writer,
                                                             timeout=aiohttp.ClientTimeout(total=None, sock_read=60))

        got_result = await self._do_read_response(queue)
        if not got_result:
            await queue.put(None)


class _LoadFromJob(Job):
    def __init__(self, location: str, params: dict | None,
                 session: _WorkerClientSession,
                 on_ready: Callable[[Job], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.location = location
        self.target_url = 'source?mode=queue&processing=sync'
        self.body = {
            "sourceType": "URL",
            "url": self.location,
        }
        if self._params is not None:
            self.body['params'] = self._params

        self.timeouts = aiohttp.ClientTimeout(total=None, sock_read=60)

    async def _do_execute_job(self, queue: Queue, session: _WorkerClientSession):
        try:
            got_result = False
            self._response = await session.pipeline_patch(self.target_url,
                                                          accept='application/jsonl',
                                                          data=json.dumps(self.body),
                                                          content_type='application/json',
                                                          timeout=self.timeouts)
            got_result = await self._do_read_response(queue)
        finally:
            if not got_result:
                await queue.put(None)

