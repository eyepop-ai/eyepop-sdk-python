import json
import logging
import mimetypes
from asyncio import Queue
from typing import Callable, BinaryIO

import aiohttp

from eyepop.worker.worker_client_session import (WorkerClientSession)
from eyepop.jobs import Job, JobStateCallback

log_requests = logging.getLogger('eyepop.requests')


class WorkerJob(Job):
    """
    Abstract Job submitted to an EyePop.ai WorkerEndpoint.
    """

    def __init__(self,
                 session: WorkerClientSession,
                 params: dict | None,
                 on_ready: Callable[["WorkerJob"], None] | None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self._params = params


    async def predict(self) -> dict:
        return await self.pop_result()

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
                    await self.push_result(prediction)
        return got_results


class _UploadJob(WorkerJob):
    def __init__(self, location: str, params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.location = location
        self.target_url = 'source?mode=queue&processing=sync'
        mime_types = mimetypes.guess_type(location)
        if len(mime_types) > 0:
            self.mime_type = mime_types[0]
        else:
            self.mime_type = 'application/octet-stream'

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
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


class _UploadStreamJob(WorkerJob):
    def __init__(self, stream: BinaryIO, mime_type: str, params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, params, on_ready, callback)
        self.stream = stream
        self.mime_type = mime_type
        self.target_url = 'source?mode=queue&processing=sync'

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
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


class _LoadFromJob(WorkerJob):
    def __init__(self, location: str, params: dict | None,
                 session: WorkerClientSession,
                 on_ready: Callable[[WorkerJob], None] | None = None,
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

    async def _do_execute_job(self, queue: Queue, session: WorkerClientSession):
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

