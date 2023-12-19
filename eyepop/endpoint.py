import asyncio
import logging
import time
from types import TracebackType
from typing import Optional, Type, Callable
from urllib.parse import urljoin
import aiohttp
from aiohttp import ClientError

from eyepop.exceptions import PopNotStartedException
from eyepop.jobs import Job, _UploadJob, _LoadFromJob
from eyepop.syncify import SyncJob

log = logging.getLogger('eyepop')


class Endpoint:
    """
    Endpoint to an EyePop.ai worker.
    """
    def __init__(self, secret_key: str, eyepop_url: str, pop_id: str, auto_start: bool,
                 stop_jobs: bool):
        self.secret_key = secret_key
        self.eyepop_url = eyepop_url
        self.pop_id = pop_id
        self.auto_start = auto_start
        self.stop_jobs = stop_jobs

        self.token = None
        self.expire_token_time = None

        self.worker_config = None

        self.session = None
        self.task_group = None

        self.tasks = set()
        self.sem = asyncio.Semaphore(1024)

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "Endpoint":
        try:
            await self.connect()
        except ClientError as e:
            await self.disconnect()
            raise e

        return self

    async def __aexit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
    ) -> None:
        await self.disconnect()

    async def __authorization_header(self):
        return f'Bearer {await self.__get_access_token()}'

    async def disconnect(self):
        tasks = list(self.tasks)
        if len(tasks) > 0:
            await asyncio.gather(*tasks)
        await self.session.close()

    async def connect(self, stop_jobs: bool = True):
        self.session = aiohttp.ClientSession(raise_for_status=True, connector=aiohttp.TCPConnector(limit=5))
        config_url = f'{self.eyepop_url}/pops/{self.pop_id}/config?auto_start={self.auto_start}'
        try:
            headers = {'Authorization': await self.__authorization_header()}
            async with self.session.get(config_url, headers=headers) as response:
                self.worker_config = await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status == 401:
                self.token = None
                self.expire_token_time = None
                headers = {'Authorization': await self.__authorization_header()}
                async with self.session.get(config_url, headers=headers) as retried_response:
                    self.worker_config = await retried_response.json()

        if self.worker_config['base_url'] is None or self.worker_config['pipeline_id'] is None:
            raise PopNotStartedException(pop_id=self.pop_id)

        if stop_jobs:
            stop_jobs_url = f'{self.__pipeline_base_url()}/source?mode=preempt&processing=sync'
            body = {'sourceType': 'NONE'}
            headers = {'Authorization': await self.__authorization_header()}
            async with self.session.patch(stop_jobs_url, headers=headers, json=body):
                response.raise_for_status()

    def __pipeline_base_url(self):
        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")
        return f'{base_url}/pipelines/{self.worker_config["pipeline_id"]}'

    async def __get_access_token(self):
        if self.token is None or self.expire_token_time < time.time():
            body = {
                'secret_key': self.secret_key
            }
            async with self.session.post(f'{self.eyepop_url}/authentication/token', json=body) as response:
                self.token = await response.json()
                self.expire_token_time = time.time() + self.token['expires_in'] - 60
        return self.token['access_token']

    def _task_done(self, task):
        self.tasks.discard(task)
        self.sem.release()

    async def upload(self, location: str, on_ready: Callable[[Job], None] | None = None) -> Job | SyncJob:
        if self.worker_config is None:
            await self.connect()
        await self.sem.acquire()
        job = _UploadJob(location=location, pipeline_base_url=self.__pipeline_base_url(),
                         authorization_header=await self.__authorization_header(), session=self.session,
                         on_ready=on_ready)
        task = asyncio.create_task(job.execute())
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return job

    async def load_from(self, location: str, on_ready: Callable[[Job], None] | None = None) -> Job | SyncJob:
        if self.worker_config is None:
            await self.connect()
        await self.sem.acquire()
        job = _LoadFromJob(location=location, pipeline_base_url=self.__pipeline_base_url(),
                           authorization_header=await self.__authorization_header(), session=self.session,
                           on_ready=on_ready)
        task = asyncio.create_task(job.execute())
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return job
