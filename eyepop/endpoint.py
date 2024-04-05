import asyncio
import logging
import time
from enum import Enum
from types import TracebackType
from typing import Optional, Type, Callable, BinaryIO
from urllib.parse import urljoin
import warnings

import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientResponse

from eyepop.exceptions import PopNotStartedException
from eyepop.jobs import Job, _UploadJob, _LoadFromJob, _JobStateCallback, _UploadStreamJob
from eyepop.syncify import SyncJob

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')
log_metrics = logging.getLogger('eyepop.metrics')


async def response_check_with_error_body(response: ClientResponse):
    if not response.ok:
        message = await response.text()
        if message is None or len(message) == 0:
            message = response.reason
        raise ClientResponseError(
            response.request_info,
            response.history,
            status=response.status,
            message=message,
            headers=response.headers,
        )


class Endpoint:
    """
    Endpoint to an EyePop.ai worker.
    """

    def __init__(self, secret_key: str, eyepop_url: str, pop_id: str, auto_start: bool,
                 stop_jobs: bool, job_queue_length: int, is_sandbox: bool):
        self.secret_key = secret_key
        self.eyepop_url = eyepop_url
        self.pop_id = pop_id
        self.auto_start = auto_start
        self.stop_jobs = stop_jobs
        self.is_sandbox = is_sandbox

        self.sandbox_id = None

        self.token = None
        self.expire_token_time = None

        self.worker_config = None

        self.session = None
        self.task_group = None

        self.tasks = set()
        self.sem = asyncio.Semaphore(job_queue_length)

        if log_metrics.getEffectiveLevel() == logging.DEBUG:
            self.metrics_collector = _MetricCollector()
        else:
            self.metrics_collector = None

        self.popComp = None

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

        if self.sandbox_id is not None:
            base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")
            delete_sandbox_url = f'{base_url}/sandboxes/{self.sandbox_id}'
            headers = {'Authorization': await self.__authorization_header()}
            log_requests.debug('before DELETE %s', delete_sandbox_url)
            async with self.session.delete(delete_sandbox_url, headers=headers) as response:
                response.raise_for_status()
            log_requests.debug('after DELETE %s', delete_sandbox_url)
            self.sandbox_id = None

        await self.session.close()
        if self.metrics_collector is not None:
            log_metrics.debug('endpoint disconnected, collected session metrics:')
            log_metrics.debug('total number of jobs: %d', self.metrics_collector.total_number_of_jobs)
            log_metrics.debug(f'max concurrent number of jobs: {self.metrics_collector.max_number_of_jobs_by_state}')
            log_metrics.debug(f'average wait time until state: {self.metrics_collector.get_average_times()}')

    async def connect(self, stop_jobs: bool = True):
        self.session = aiohttp.ClientSession(raise_for_status=response_check_with_error_body,
                                             connector=aiohttp.TCPConnector(limit=5))
        if self.pop_id == 'transient':
            config_url = f'{self.eyepop_url}/workers/config'
        else:
            config_url = f'{self.eyepop_url}/pops/{self.pop_id}/config?auto_start={self.auto_start}'
        headers = {'Authorization': await self.__authorization_header()}
        try:
            log_requests.debug('before GET %s', config_url)
            async with self.session.get(config_url, headers=headers) as response:
                self.worker_config = await response.json()
        except aiohttp.ClientResponseError as e:
            if e.status != 401:
                raise e
            else:
                log_requests.debug('after GET %s: 401, about to retry with fresh access token', config_url)
                self.token = None
                self.expire_token_time = None
                headers = {'Authorization': await self.__authorization_header()}
                async with self.session.get(config_url, headers=headers) as retried_response:
                    self.worker_config = await retried_response.json()

        log_requests.debug(f'after GET {config_url}: {self.worker_config}')

        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")

        if self.is_sandbox:
            create_sandbox_url = f'{base_url}/sandboxes'
            headers = {'Authorization': await self.__authorization_header()}
            log_requests.debug('before POST %s', create_sandbox_url)
            async with self.session.post(create_sandbox_url, headers=headers) as response:
                response.raise_for_status()
                response_json = await response.json()
            log_requests.debug('after POST %s', create_sandbox_url)
            self.sandbox_id = response_json

        if self.pop_id == 'transient':
            self.popComp = 'identity'

            if self.sandbox_id is None:
                start_pipeline_url = f'{base_url}/pipelines'
            else:
                start_pipeline_url = f'{base_url}/pipelines?sandboxId={self.sandbox_id}'

            body = {
                'inferPipelineDef': {
                    'pipeline': self.popComp
                },
                "source": {
                    "sourceType": "NONE",
                },
                "idleTimeoutSeconds": 60,
                "logging": ["out_meta"],
                "videoOutput": "no_output"
            }
            headers = {'Authorization': await self.__authorization_header()}
            log_requests.debug('before POST %s', start_pipeline_url)
            async with self.session.post(start_pipeline_url, headers=headers, json=body) as response:
                response.raise_for_status()
                response_json = await response.json()
            log_requests.debug('after POST %s', start_pipeline_url)
            self.worker_config['pipeline_id'] = response_json['id']

        if self.worker_config['base_url'] is None or self.worker_config['pipeline_id'] is None:
            raise PopNotStartedException(pop_id=self.pop_id)

        if stop_jobs:
            stop_jobs_url = f'{self.__pipeline_base_url()}/source?mode=preempt&processing=sync'
            body = {'sourceType': 'NONE'}
            headers = {'Authorization': await self.__authorization_header()}
            log_requests.debug('before PATCH %s', stop_jobs_url)
            async with self.session.patch(stop_jobs_url, headers=headers, json=body) as response:
                response.raise_for_status()
            log_requests.debug('after PATCH %s', stop_jobs_url)

        if self.popComp is None:
            # get current pipeline string and store
            get_url = f'{self.__pipeline_base_url()}'
            headers = {'Authorization': await self.__authorization_header()}
            log_requests.debug('before GET %s', get_url)
            async with self.session.get(get_url, headers=headers) as response:
                response.raise_for_status()
                response_json = await response.json()
                self.popComp = response_json['inferPipeline']
            log_requests.debug('after GET %s', get_url)
            log_requests.debug('current popComp is %s', self.popComp)

    async def get_pop_comp(self) -> str:
        if self.worker_config is None:
            await self.connect()
        return self.popComp
    
    async def set_pop_comp(self, popComp: str = None):
        if self.worker_config is None:
            await self.connect()
        headers = {
            'Authorization': await self.__authorization_header(),
            'Content-Type': 'application/json'
        }
        body = {
            'pipeline': popComp
        }
        patch_url = f'{self.__pipeline_base_url()}/inferencePipeline'
        log_requests.debug('before PATCH %s', patch_url)
        async with self.session.patch(patch_url, headers=headers, json=body) as response:
            response.raise_for_status()
        log_requests.debug('after PATCH %s', patch_url)
        self.popComp = popComp
    
    '''
    Deprecated
    '''
    async def list_models(self) -> dict:
        warnings.warn("list_models for development use only", DeprecationWarning)
        if self.worker_config is None:
            await self.connect()
        headers = {
            'Authorization': await self.__authorization_header(),
            'Content-Type': 'application/json'
        }

        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")
        if self.sandbox_id is None:
            get_url = f'{base_url}/models/instances'
        else:
            get_url = f'{base_url}/models/instances?sandboxId={self.sandbox_id}'

        log_requests.debug('before GET %s', get_url)
        async with self.session.get(get_url, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_manifest(self) -> dict:
        warnings.warn("list_models for development use only", DeprecationWarning)
        if self.worker_config is None:
            await self.connect()
        headers = {
            'Authorization': await self.__authorization_header(),
        }
        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")

        if self.sandbox_id is None:
            get_url = f'{base_url}/models/sources'
        else:
            get_url = f'{base_url}/models/sources?sandboxId={self.sandbox_id}'

        log_requests.debug('before GET %s', get_url)
        async with self.session.get(get_url, headers=headers) as response:
            response.raise_for_status()
            return await response.json()
    
    async def set_manifest(self, manifest: dict):
        warnings.warn("list_models for development use only", DeprecationWarning)
        if self.worker_config is None:
            await self.connect()
        headers = {
            'Authorization': await self.__authorization_header(),
            'Content-Type': 'application/json'
        }
        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")

        if self.sandbox_id is None:
            put_url = f'{base_url}/models/sources'
        else:
            put_url = f'{base_url}/models/sources?sandboxId={self.sandbox_id}'

        log_requests.debug('before PUT %s', put_url)
        async with self.session.put(put_url, headers=headers, json=manifest) as response:
            response.raise_for_status()
        log_requests.debug('after PUT %s', put_url)

    async def load_model(self, model:dict, override:bool = False):
        warnings.warn("list_models for development use only", DeprecationWarning)
        if override:
            await self.purge_model(model)
        if self.worker_config is None:
            await self.connect()
        headers = {
            'Authorization': await self.__authorization_header(),
            'Content-Type': 'application/json'
        }
        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")

        if self.sandbox_id is None:
            post_url = f'{base_url}/models/instances'
        else:
            post_url = f'{base_url}/models/instances?sandboxId={self.sandbox_id}'

        log_requests.debug('before POST %s', post_url)
        async with self.session.post(post_url, headers=headers, json=model) as response:
            response.raise_for_status()
        log_requests.debug('after POST %s', post_url)

    async def purge_model(self, model:dict):
        warnings.warn("list_models for development use only", DeprecationWarning)
        headers = {
            'Authorization': await self.__authorization_header(),
            'Content-Type': 'application/json'
        }
        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")

        if self.sandbox_id is None:
            delete_url = f'{base_url}/models/instances'
        else:
            delete_url = f'{base_url}/models/instances?sandboxId={self.sandbox_id}'

        log_requests.debug('before DELETE %s', delete_url)
        async with self.session.delete(delete_url, headers=headers, json=model) as response:
            response.raise_for_status()
    '''
    '''

    def __pipeline_base_url(self):
        base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")
        return f'{base_url}/pipelines/{self.worker_config["pipeline_id"]}'

    async def __get_access_token(self):
        now = time.time()
        if self.token is None or self.expire_token_time < now:
            body = {
                'secret_key': self.secret_key
            }
            post_url = f'{self.eyepop_url}/authentication/token'
            log_requests.debug('before POST %s', post_url)
            async with self.session.post(post_url, json=body) as response:
                self.token = await response.json()
                self.expire_token_time = time.time() + self.token['expires_in'] - 60
            log_requests.debug('after POST %s expires_in=%d token_type=%s', post_url, self.token['expires_in'],
                               self.token['token_type'])
        log.debug('using access token, valid for at least %d seconds', self.expire_token_time - now)
        return self.token['access_token']

    def _task_done(self, task):
        self.tasks.discard(task)
        self.sem.release()

    async def upload(self, location: str, params: dict | None = None, on_ready: Callable[[Job], None] | None = None) -> Job | SyncJob:
        if self.worker_config is None:
            await self.connect()
        await self.sem.acquire()
        job = _UploadJob(location=location, params=params, pipeline_base_url=self.__pipeline_base_url(),
                         authorization_header=await self.__authorization_header(), session=self.session,
                         on_ready=on_ready, callback=self.metrics_collector)
        task = asyncio.create_task(job.execute())
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return job

    async def upload_stream(self, stream: BinaryIO, mime_type: str, params: dict | None = None, on_ready: Callable[[Job], None] | None = None) -> Job | SyncJob:
        if self.worker_config is None:
            await self.connect()
        await self.sem.acquire()
        job = _UploadStreamJob(stream, mime_type, params=params, pipeline_base_url=self.__pipeline_base_url(),
                         authorization_header=await self.__authorization_header(), session=self.session,
                         on_ready=on_ready, callback=self.metrics_collector)
        task = asyncio.create_task(job.execute())
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return job

    async def load_from(self, location: str, params: dict | None = None, on_ready: Callable[[Job], None] | None = None) -> Job | SyncJob:
        if self.worker_config is None:
            await self.connect()
        await self.sem.acquire()
        job = _LoadFromJob(location=location, params=params, pipeline_base_url=self.__pipeline_base_url(),
                           authorization_header=await self.__authorization_header(), session=self.session,
                           on_ready=on_ready, callback=self.metrics_collector)
        task = asyncio.create_task(job.execute())
        self.tasks.add(task)
        task.add_done_callback(self._task_done)
        return job


class JobState(Enum):
    CREATED = 1
    STARTED = 2
    IN_PROGRESS = 3
    FINISHED = 4
    FAILED = 5
    DRAINED = 6

    def __repr__(self):
        return self._name_


class _MetricCollector(_JobStateCallback):
    def __init__(self):
        self.jobs_to_state = {}
        self.jobs_to_last_updated = {}
        self.total_number_of_jobs = 0
        self.max_number_of_jobs_by_state = {
            JobState.STARTED: 0,
            JobState.IN_PROGRESS: 0,
            JobState.FINISHED: 0,
            JobState.DRAINED: 0,
            JobState.FAILED: 0,
        }
        self.number_of_jobs_reached_state = {
            JobState.IN_PROGRESS: 0,
            JobState.FINISHED: 0,
            JobState.DRAINED: 0,
            JobState.FAILED: 0,
        }
        self.total_time_to_reached_state = {
            JobState.IN_PROGRESS: 0.0,
            JobState.FINISHED: 0.0,
            JobState.DRAINED: 0.0,
            JobState.FAILED: 0.0,
        }

    def get_average_time(self, state: JobState) -> float:
        if self.number_of_jobs_reached_state[state] == 0:
            return 0.0
        else:
            return self.total_time_to_reached_state[state] / self.number_of_jobs_reached_state[state]

    def get_average_times(self) -> dict:
        times = {
            JobState.IN_PROGRESS: self.get_average_time(JobState.IN_PROGRESS),
            JobState.FINISHED: self.get_average_time(JobState.FINISHED),
            JobState.DRAINED: self.get_average_time(JobState.DRAINED),
            JobState.FAILED: self.get_average_time(JobState.FAILED),
        }
        return times

    def update_count_by_state(self, state: JobState):
        count = 0
        for job, job_state in self.jobs_to_state.items():
            if state == job_state:
                count += 1
        if count > self.max_number_of_jobs_by_state[state]:
            self.max_number_of_jobs_by_state[state] = count

    def collect_execution_time(self, job: Job, new_state: JobState):
        now = time.time()
        duration = now - self.jobs_to_last_updated[job]
        self.jobs_to_last_updated[job] = now
        self.number_of_jobs_reached_state[new_state] += 1
        self.total_time_to_reached_state[new_state] += duration

    def created(self, job):
        self.total_number_of_jobs += 1
        self.jobs_to_state[job] = JobState.CREATED
        self.jobs_to_last_updated[job] = time.time()

    def started(self, job):
        current_state = self.jobs_to_state[job]
        if current_state != JobState.CREATED:
            log.debug("invalid job state %v in metrics collector for started(%v)", current_state, job)
        else:
            self.jobs_to_state[job] = JobState.STARTED
        self.jobs_to_last_updated[job] = time.time()
        self.update_count_by_state(JobState.STARTED)

    def first_result(self, job):
        current_state = self.jobs_to_state[job]
        if current_state != JobState.STARTED:
            log.debug("invalid job state %v in metrics collector for first_result(%v)", current_state, job)
        else:
            self.jobs_to_state[job] = JobState.IN_PROGRESS
        self.collect_execution_time(job, JobState.IN_PROGRESS)
        self.update_count_by_state(JobState.IN_PROGRESS)

    def failed(self, job):
        self.jobs_to_state[job] = JobState.FAILED
        self.collect_execution_time(job, JobState.FAILED)
        self.update_count_by_state(JobState.FAILED)
        self.jobs_to_last_updated[job] = time.time()

    def finished(self, job):
        self.jobs_to_state[job] = JobState.FINISHED
        self.collect_execution_time(job, JobState.FINISHED)
        self.update_count_by_state(JobState.FINISHED)
        self.jobs_to_last_updated[job] = time.time()

    def drained(self, job):
        self.jobs_to_state[job] = JobState.DRAINED
        self.collect_execution_time(job, JobState.DRAINED)
        self.update_count_by_state(JobState.DRAINED)
        self.jobs_to_last_updated[job] = time.time()

    def finalized(self, job):
        del self.jobs_to_state[job]
