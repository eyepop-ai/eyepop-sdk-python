import logging
import time
from io import StringIO
from typing import Any, BinaryIO, Callable
from urllib.parse import urljoin

import aiohttp
from aiohttp.client import _RequestContextManager

from eyepop.compute.api import fetch_session_endpoint
from eyepop.endpoint import Endpoint
from eyepop.exceptions import (
    PopConfigurationException,
    PopNotReachableException,
    PopNotStartedException,
)
from eyepop.settings import settings
from eyepop.worker.load_balancer import EndpointLoadBalancer
from eyepop.worker.worker_client_session import WorkerClientSession
from eyepop.worker.worker_jobs import (
    WorkerJob,
    _LoadFromAssetUuidJob,
    _LoadFromJob,
    _UploadFileJob,
    _UploadStreamJob,
)
from eyepop.worker.worker_syncify import SyncWorkerJob
from eyepop.worker.worker_types import ComponentParams, Pop, VideoMode

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')
log_metrics = logging.getLogger('eyepop.metrics')


def should_use_compute_api(pop_id: str, api_key: str | None) -> bool:
    """Determine if we should use Compute API based on pop_id and api_key."""
    if not api_key:
        return False

    is_transient = pop_id == "transient" or not pop_id

    if not is_transient:
        log.debug(f"Pop ID {pop_id} is not transient, will not use compute API")
        return False

    log.debug("Using compute API")
    return True

class WorkerEndpoint(Endpoint, WorkerClientSession):
    """Endpoint to an EyePop.ai worker."""

    def __init__(
            self,
            secret_key: str | None,
            access_token: str | None,
            api_key: str | None,
            eyepop_url: str,
            pop_id: str,
            auto_start: bool,
            stop_jobs: bool,
            job_queue_length: int,
            request_tracer_max_buffer: int,
            dataset_uuid: str | None = None,
    ):
        super().__init__(
            secret_key=secret_key,
            access_token=access_token,
            eyepop_url=eyepop_url,
            api_key=api_key,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
        )
        self.pop_id = pop_id
        self.auto_start = auto_start
        self.stop_jobs = stop_jobs
        self.dataset_uuid = dataset_uuid

        self.is_dev_mode = True

        self.worker_config = None
        self.last_fetch_config_success_time = None
        self.last_fetch_config_error = None
        self.last_fetch_config_error_time = None

        # new way
        self.pop = Pop(components=[])

        self.add_retry_handler(404, self._retry_404)

    async def _retry_404(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('after 404, about to retry with fresh config')
            self.worker_config = None
            return True

    async def _disconnect(self, timeout: float | None = None):
        client_timeout = None
        if timeout is not None:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
        if (self.is_dev_mode and self.pop_id == 'transient'
                and self.worker_config is not None and self.worker_config.get('pipeline_id') is not None
                and self.client_session is not None):
            try:
                base_url = await self.dev_mode_base_url()
                delete_pipeline_url = f'{base_url}/pipelines/{self.worker_config["pipeline_id"]}'
                headers = {}
                authorization_header = await self._authorization_header()
                if authorization_header is not None:
                    headers['Authorization'] = authorization_header
                await self.client_session.delete(delete_pipeline_url, headers=headers, timeout=client_timeout)
            except Exception as e:
                log.exception(e, exc_info=True)
            finally:
                del self.worker_config["pipeline_id"]

    async def _reconnect(self):
        if self.worker_config is not None:
            return

        if self.last_fetch_config_error_time is not None and self.last_fetch_config_error_time > time.time() - settings.min_config_reconnect_secs:
            raise self.last_fetch_config_error

        if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time > time.time() - settings.min_config_reconnect_secs:
            raise aiohttp.ClientConnectionError()

        if self.compute_ctx:
            if not self.compute_ctx.session_endpoint:
                log_requests.debug("Fetching compute API session")
                self.compute_ctx = await fetch_session_endpoint(self.compute_ctx, self.client_session)
                self.eyepop_url = self.compute_ctx.session_endpoint
                log_requests.debug(f"Compute session ready: {self.compute_ctx.session_endpoint}")

            self.worker_config = {
                "session_endpoint": self.compute_ctx.session_endpoint,
                "pipeline_id": self.compute_ctx.pipeline_uuid,
                "endpoints": []
            }

            self.last_fetch_config_success_time = time.time()
            self.last_fetch_config_error = None
            self.last_fetch_config_error_time = None
        else:
            if self.pop_id == 'transient':
                config_url = f'{self.eyepop_url}/workers/config'
            else:
                config_url = f'{self.eyepop_url}/pops/{self.pop_id}/config?auto_start={self.auto_start}'
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers['Authorization'] = authorization_header
            try:
                async with self.client_session.get(config_url, headers=headers) as response:
                    self.worker_config = await response.json()
                self.last_fetch_config_success_time = time.time()
                self.last_fetch_config_error = None
                self.last_fetch_config_error_time = None
            except aiohttp.ClientResponseError as e:
                if e.status != 401:
                    self.last_fetch_config_error = e
                    self.last_fetch_config_error_time = time.time()
                    raise e
                else:
                    self.token = None
                    self.expire_token_time = None
                    headers = {}
                    authorization_header = await self._authorization_header()
                    if authorization_header is not None:
                        headers['Authorization'] = authorization_header
                    async with self.client_session.get(config_url, headers=headers) as retried_response:
                        self.worker_config = await retried_response.json()
            except aiohttp.ClientConnectionError as e:
                self.last_fetch_config_error = e
                self.last_fetch_config_error_time = time.time()
                raise e

        self.is_dev_mode = self.pop_id == 'transient' or self.worker_config.get('status') != 'active_prod'

        if self.compute_ctx:
            log_requests.debug(f'Using compute context config: {self.worker_config}')

        base_url = await self.dev_mode_base_url()

        if self.pop_id == 'transient' or (self.compute_ctx and (not self.compute_ctx.pipeline_uuid or self.compute_ctx.pipeline_uuid == "")):
            start_pipeline_url = f'{base_url}/pipelines'
            body = {
                "pop": self.pop.model_dump() if self.pop else {},
                "source": {
                    "sourceType": "NONE"
                },
                "idleTimeoutSeconds": 60,
                "logging": ["out_meta"],
                "videoOutput": "no_output",
            }
            if self.dataset_uuid is not None:
                body["datasetUuid"] = self.dataset_uuid

            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers['Authorization'] = authorization_header
            async with self.client_session.post(start_pipeline_url, headers=headers, json=body) as response:
                response_json = await response.json()
            self.worker_config['pipeline_id'] = response_json['id']

            if self.compute_ctx:
                self.compute_ctx.pipeline_uuid = response_json['id']
                logging.info(f"Created pipeline with ID: {response_json['id']}")

            if self.is_dev_mode:
                if 'session_endpoint' in self.worker_config:
                    base_url = self.worker_config['session_endpoint'].rstrip("/")
                else:
                    base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")
                endpoint = {'base_url': base_url, 'pipeline_id': self.worker_config['pipeline_id']}
                self.load_balancer = EndpointLoadBalancer([endpoint])
                log.debug(f"Reinitialized load balancer with pipeline_id: {self.worker_config['pipeline_id']}")

        if self.is_dev_mode:
            has_base_url = ('base_url' in self.worker_config and self.worker_config['base_url'] is not None) or \
                          ('session_endpoint' in self.worker_config and self.worker_config['session_endpoint'] is not None)
            has_pipeline_id = self.worker_config.get('pipeline_id') is not None
            if not has_base_url or not has_pipeline_id:
                raise PopNotStartedException(pop_id=self.pop_id)

        if self.is_dev_mode and self.stop_jobs:
            stop_jobs_url = f'{await self.dev_mode_pipeline_base_url()}/source?mode=preempt&processing=sync'
            body = {'sourceType': 'NONE'}
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers['Authorization'] = authorization_header
            async with self.client_session.patch(stop_jobs_url, headers=headers, json=body) as response:
                pass

        if self.is_dev_mode and self.pop is None:
            get_url = await self.dev_mode_pipeline_base_url()
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers['Authorization'] = authorization_header
            async with self.client_session.get(get_url, headers=headers) as response:
                response_json = await response.json()
                pop_as_dict = response_json.get('pop')
                if pop_as_dict is None:
                    self.pop = None
                else:
                    self.pop = Pop(**pop_as_dict)
            log.debug('current pop is %s', self.pop)

        if self.is_dev_mode:
            if 'session_endpoint' in self.worker_config:
                base_url = self.worker_config['session_endpoint'].rstrip("/")
            else:
                base_url = urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")
            endpoint = {'base_url': base_url, 'pipeline_id': self.worker_config['pipeline_id']}
            self.load_balancer = EndpointLoadBalancer([endpoint])
            log.info(f"Initialized load balancer with endpoint: {endpoint}")
        else:
            self.load_balancer = EndpointLoadBalancer(self.worker_config['endpoints'])
            log.info(f"Initialized load balancer with endpoints: {self.worker_config['endpoints']}")


    async def session(self) -> dict:
        session = await super().session()
        session['popId'] = self.pop_id
        if self.worker_config:
            if 'session_endpoint' in self.worker_config:
                session['baseUrl'] = self.worker_config['session_endpoint']
            elif 'base_url' in self.worker_config:
                session['baseUrl'] = self.worker_config['base_url']
            session['pipelineId'] = self.worker_config['pipeline_id']
        return session

    async def get_pop(self) -> Pop | None:
        return self.pop

    async def set_pop(self, pop: Pop):
        if not self.is_dev_mode:
            raise PopConfigurationException(self.pop_id, 'set_pop not supported in production mode')
        response = await self.pipeline_patch('pop', content_type='application/json',
                                             data=pop.model_dump_json())
        self.pop = pop
        return response

    async def dev_mode_pipeline_base_url(self):
        if self.is_dev_mode:
            return f'{await self.dev_mode_base_url()}/pipelines/{self.worker_config["pipeline_id"]}'
        else:
            pass

    async def upload(
            self,
            location: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob | SyncWorkerJob:
        job = _UploadFileJob(
            location=location,
            video_mode=video_mode,
            component_params=params,
            session=self, on_ready=on_ready,
            callback=self.metrics_collector
        )
        await  self._task_start(job.execute())
        return job

    async def upload_stream(
            self,
            stream: BinaryIO,
            mime_type: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob | SyncWorkerJob:
        job = _UploadStreamJob(
            stream=stream,
            mime_type=mime_type,
            video_mode=video_mode,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector
        )
        await self._task_start(job.execute())
        return job

    async def load_from(
            self,
            location: str,
            params: list[ComponentParams] | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob | SyncWorkerJob:
        job = _LoadFromJob(
            location=location,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector
        )
        await self._task_start(job.execute())
        return job

    async def load_asset(
            self,
            asset_uuid: str,
            params: list[ComponentParams] | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob | SyncWorkerJob:
        job = _LoadFromAssetUuidJob(
            asset_uuid=asset_uuid,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector
        )
        await self._task_start(job.execute())
        return job

    async def dev_mode_base_url(self) -> str:
        if self.worker_config is None:
            await self._reconnect()
        if 'session_endpoint' in self.worker_config:
            return self.worker_config['session_endpoint'].rstrip("/")
        return urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/")

    #
    # Implements: _WorkerClientSession
    #

    async def pipeline_get(self, url_path_and_query: str, accept: str | None = None,
                           timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        return await self._pipeline_request_with_retry('GET', url_path_and_query, accept=accept, timeout=timeout)

    async def pipeline_post(self, url_path_and_query: str, accept: str | None = None, open_data: Callable = None,
                            content_type: str | None = None,
                            timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        return await self._pipeline_request_with_retry('POST', url_path_and_query, accept=accept, open_data=open_data,
                                                       content_type=content_type, timeout=timeout)

    async def pipeline_patch(self, url_path_and_query: str, accept: str | None = None, data: Any = None,
                             content_type: str | None = None,
                             timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        def open_data():
            if isinstance(data, str):
                return StringIO(data)
            else:
                return data

        return await self._pipeline_request_with_retry('PATCH', url_path_and_query, accept=accept, open_data=open_data,
                                                       content_type=content_type, timeout=timeout)

    async def pipeline_delete(self, url_path_and_query: str,
                              timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        return await self._pipeline_request_with_retry('DELETE', url_path_and_query, timeout=timeout)

    async def _pipeline_request_with_retry(self, method: str, url_path_and_query: str, accept: str | None = None,
                                           open_data: Callable = None, content_type: str | None = None,
                                           timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time < time.time() - settings.force_refresh_config_secs:
            self.worker_config = None

        failed_attempts = 0
        retried_re_config = False
        start_time = time.time()

        while time.time() - start_time < settings.max_retry_time_secs:
            if self.worker_config is None:
                await self._reconnect()

            entry = self.load_balancer.next_entry(settings.max_retry_time_secs + 1)
            log.debug(f"Load balancer entry: {entry}")
            if entry is None:
                if not retried_re_config:
                    # pipeline might have just shut down
                    log_requests.debug('no healthy endpoints, about to retry with fresh config')
                    self.worker_config = None
                    retried_re_config = True
                    continue
                else:
                    raise PopNotReachableException(self.pop_id, self.load_balancer.get_debug_status())

            url = f'{entry.base_url}/pipelines/{entry.pipeline_id}/{url_path_and_query}'

            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers['Authorization'] = authorization_header
            if accept is not None:
                headers['Accept'] = accept
            if content_type is not None:
                headers['Content-Type'] = content_type
            try:
                if open_data is not None:
                    with open_data() as data:
                        if isinstance(data, StringIO):
                            response = await self.client_session.request(method, url, headers=headers,
                                                                         data=data.getvalue(), timeout=timeout)
                        else:
                            response = await self.client_session.request(method, url, headers=headers, data=data,
                                                                         timeout=timeout)
                else:
                    response = await self.client_session.request(method, url, headers=headers, timeout=timeout)

                entry.mark_success()

                return response
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    # in load balanced configuration, we overwrite the standard 404 handler
                    entry.mark_error()
                else:
                    failed_attempts += 1
                    if e.status not in self.retry_handlers:
                        log_requests.exception('unexpected error: %s', e)
                        raise e
                    if not await self.retry_handlers[e.status](e.status, failed_attempts):
                        log_requests.exception('unexpected error')
                        raise e
            except aiohttp.ClientConnectionError:
                entry.mark_error()
            except Exception as e:
                log_requests.exception('unexpected error')
                raise e

        raise PopNotReachableException(self.pop_id, [])

    async def _worker_request_with_retry(self, method: str, url_path_and_query: str, accept: str | None = None,
                                         data: Any = None, content_type: str | None = None,
                                         timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        url = f'{await self.dev_mode_base_url()}/{url_path_and_query}'
        return await self.request_with_retry(method, url, accept, data, content_type, timeout)
