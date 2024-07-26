import json
import logging
import time
from io import StringIO
from typing import Callable, BinaryIO, Any
from urllib.parse import urljoin
import warnings

import aiohttp

from eyepop.endpoint import Endpoint
from eyepop.exceptions import PopNotStartedException, PopConfigurationException, PopNotReachableException
from eyepop.worker.worker_jobs import WorkerJob, _UploadJob, _LoadFromJob, _UploadStreamJob
from eyepop.worker.load_balancer import EndpointLoadBalancer
from eyepop.worker.worker_syncify import SyncWorkerJob

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')
log_metrics = logging.getLogger('eyepop.metrics')

MIN_CONFIG_RECONNECT_SECS = 10.0
MAX_RETRY_TIME_SECS = 30.0
FORCE_REFRESH_CONFIG_SECS = (61.0 * 61.0)


class WorkerEndpoint(Endpoint):
    """
    Endpoint to an EyePop.ai worker.
    """

    def __init__(self, secret_key: str, eyepop_url: str, pop_id: str, auto_start: bool, stop_jobs: bool,
                 job_queue_length: int, is_sandbox: bool, request_tracer_max_buffer: int):
        super().__init__(secret_key, eyepop_url, job_queue_length, request_tracer_max_buffer)
        self.pop_id = pop_id
        self.auto_start = auto_start
        self.stop_jobs = stop_jobs
        self.is_sandbox = is_sandbox

        self.sandbox_id = None

        self.is_dev_mode = True

        self.worker_config = None
        self.last_fetch_config_success_time = None
        self.last_fetch_config_error = None
        self.last_fetch_config_error_time = None

        self.pop_comp = None
        self.post_transform = None
        self.add_retry_handler(404, self._retry_404)

    async def _retry_404(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('after 404, about to retry with fresh config')
            self.worker_config = None
            return True

    async def _disconnect(self):
        if self.sandbox_id is not None:
            try:
                base_url = await self.dev_mode_base_url()
                delete_sandbox_url = f'{base_url}/sandboxes/{self.sandbox_id}'
                headers = {'Authorization': await self._authorization_header()}
                log_requests.debug('before DELETE %s', delete_sandbox_url)
                await self.client_session.delete(delete_sandbox_url, headers=headers)
                log_requests.debug('after DELETE %s', delete_sandbox_url)
            except Exception as e:
                log.exception("error at disconnect", e)
            finally:
                self.sandbox_id = None

    async def _reconnect(self):
        if self.worker_config is not None:
            return

        if self.last_fetch_config_error_time is not None and self.last_fetch_config_error_time > time.time() - MIN_CONFIG_RECONNECT_SECS:
            raise self.last_fetch_config_error

        if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time > time.time() - MIN_CONFIG_RECONNECT_SECS:
            raise aiohttp.ClientConnectionError()

        if self.pop_id == 'transient':
            config_url = f'{self.eyepop_url}/workers/config'
        else:
            config_url = f'{self.eyepop_url}/pops/{self.pop_id}/config?auto_start={self.auto_start}'
        headers = {'Authorization': await self._authorization_header()}
        try:
            log_requests.debug('before GET %s', config_url)
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
                log_requests.debug('after GET %s: 401, about to retry with fresh access token', config_url)
                self.token = None
                self.expire_token_time = None
                headers = {'Authorization': await self._authorization_header()}
                async with self.client_session.get(config_url, headers=headers) as retried_response:
                    self.worker_config = await retried_response.json()
        except aiohttp.ClientConnectionError as e:
            self.last_fetch_config_error = e
            self.last_fetch_config_error_time = time.time()
            raise e

        self.is_dev_mode = self.is_sandbox or self.pop_id == 'transient' or self.worker_config[
            'status'] != 'active_prod'

        log_requests.debug(f'after GET {config_url}: {self.worker_config}')

        base_url = await self.dev_mode_base_url()

        if self.is_sandbox and self.sandbox_id is None:
            create_sandbox_url = f'{base_url}/sandboxes'
            headers = {'Authorization': await self._authorization_header()}
            log_requests.debug('before POST %s', create_sandbox_url)
            async with self.client_session.post(create_sandbox_url, headers=headers) as response:
                response_json = await response.json()
            log_requests.debug('after POST %s', create_sandbox_url)
            self.sandbox_id = response_json

        if self.pop_id == 'transient':
            self.pop_comp = 'identity'
            if self.sandbox_id is None:
                start_pipeline_url = f'{base_url}/pipelines'
            else:
                start_pipeline_url = f'{base_url}/pipelines?sandboxId={self.sandbox_id}'

            body = {'inferPipelineDef': {'pipeline': self.pop_comp},
                    'postTransformDef': {'transform': self.post_transform}, "source": {"sourceType": "NONE", },
                    "idleTimeoutSeconds": 60, "logging": ["out_meta"], "videoOutput": "no_output"}

            headers = {'Authorization': await self._authorization_header()}
            log_requests.debug('before POST %s', start_pipeline_url)
            async with self.client_session.post(start_pipeline_url, headers=headers, json=body) as response:
                response_json = await response.json()
            log_requests.debug('after POST %s', start_pipeline_url)
            self.worker_config['pipeline_id'] = response_json['id']

        if self.is_dev_mode and (self.worker_config['base_url'] is None or self.worker_config['pipeline_id'] is None):
            raise PopNotStartedException(pop_id=self.pop_id)

        if self.is_dev_mode and self.stop_jobs:
            stop_jobs_url = f'{await self.dev_mode_pipeline_base_url()}/source?mode=preempt&processing=sync'
            body = {'sourceType': 'NONE'}
            headers = {'Authorization': await self._authorization_header()}
            log_requests.debug('before PATCH %s', stop_jobs_url)
            async with self.client_session.patch(stop_jobs_url, headers=headers, json=body) as response:
                pass
            log_requests.debug('after PATCH %s', stop_jobs_url)

        if self.is_dev_mode and self.pop_comp is None:
            # get current pipeline string and store
            get_url = await self.dev_mode_pipeline_base_url()
            headers = {'Authorization': await self._authorization_header()}
            log_requests.debug('before GET %s', get_url)
            async with self.client_session.get(get_url, headers=headers) as response:
                response_json = await response.json()
                self.pop_comp = response_json.get('inferPipeline')
                self.post_transform = response_json.get('postTransform')
            log_requests.debug('after GET %s', get_url)
            log.debug('current popComp is %s', self.pop_comp)

        if self.is_dev_mode:
            endpoint = {'base_url': urljoin(self.eyepop_url, self.worker_config['base_url']).rstrip("/"),
                'pipeline_id': self.worker_config['pipeline_id']}
            self.load_balancer = EndpointLoadBalancer([endpoint])
        else:
            self.load_balancer = EndpointLoadBalancer(self.worker_config['endpoints'])

    async def get_pop_comp(self) -> str:
        return self.pop_comp

    async def set_pop_comp(self, pop_comp: str = None):
        if not self.is_dev_mode:
            raise PopConfigurationException(self.pop_id, 'set_pop_comp not supported in production mode')
        response = await self.pipeline_patch('inferencePipeline', content_type='application/json',
                                             data=json.dumps({'pipeline': pop_comp}))
        self.pop_comp = pop_comp
        return response

    async def get_post_transform(self) -> str:
        return self.post_transform

    async def set_post_transform(self, transform: str = None):
        if not self.is_dev_mode:
            raise PopConfigurationException(self.pop_id, 'set_post_transform not supported in production mode')
        response = await self.pipeline_patch('postTransform', content_type='application/json',
                                             data=json.dumps({'transform': transform}))
        self.post_transform = transform
        return response

    '''
    Deprecated
    '''

    async def list_models(self) -> list[dict]:
        warnings.warn("list_models for development use only", DeprecationWarning)
        if self.sandbox_id is None:
            get_path = 'models/instances'
        else:
            get_path = f'models/instances?sandboxId={self.sandbox_id}'

        response = await self._worker_request_with_retry('GET', url_path_and_query=get_path)
        return await response.json()

    async def get_manifest(self) -> list[dict]:
        warnings.warn("get_manifest for development use only", DeprecationWarning)
        if self.sandbox_id is None:
            get_path = 'models/sources'
        else:
            get_path = f'models/sources?sandboxId={self.sandbox_id}'
        response = await self._worker_request_with_retry('GET', url_path_and_query=get_path)
        return await response.json()

    async def set_manifest(self, manifests: list[dict]):
        if not self.is_dev_mode:
            raise PopConfigurationException(self.pop_id, 'set_manifest not supported in production mode')
        warnings.warn("set_manifest for development use only", DeprecationWarning)
        if self.sandbox_id is None:
            put_path = 'models/sources'
        else:
            put_path = f'models/sources?sandboxId={self.sandbox_id}'
        await self._worker_request_with_retry('PUT', url_path_and_query=put_path, content_type='application/json',
                                              data=json.dumps(manifests))

    async def load_model(self, model: dict, override: bool = False) -> dict:
        if not self.is_dev_mode:
            raise PopConfigurationException(self.pop_id, 'load_model not supported in production mode')
        warnings.warn("load_model for development use only", DeprecationWarning)
        if override:
            await self.unload_model(model)
        if self.sandbox_id is None:
            post_path = f'models/instances'
        else:
            post_path = f'models/instances?sandboxId={self.sandbox_id}'
        response = await self._worker_request_with_retry('POST', url_path_and_query=post_path,
                                                         content_type='application/json', data=json.dumps(model))
        return await response.json()

    async def unload_model(self, model_id: str):
        if not self.is_dev_mode:
            raise PopConfigurationException(self.pop_id, 'unload_model not supported in production mode')
        warnings.warn("purge_model for development use only", DeprecationWarning)
        if self.sandbox_id is None:
            delete_path = 'models/instances/{model["id"]}'
        else:
            delete_path = f'models/instances/{model_id}?sandboxId={self.sandbox_id}'

        await self._worker_request_with_retry('DELETE', url_path_and_query=delete_path)

    '''
    '''

    async def dev_mode_pipeline_base_url(self):
        if self.is_dev_mode:
            return f'{await self.dev_mode_base_url()}/pipelines/{self.worker_config["pipeline_id"]}'
        else:
            pass

    async def upload(self, location: str, params: dict | None = None,
                     on_ready: Callable[[WorkerJob], None] | None = None) -> WorkerJob | SyncWorkerJob:
        job = _UploadJob(location=location, params=params, session=self, on_ready=on_ready,
                         callback=self.metrics_collector)
        await  self._task_start(job.execute())
        return job

    async def upload_stream(self, stream: BinaryIO, mime_type: str, params: dict | None = None,
                            on_ready: Callable[[WorkerJob], None] | None = None) -> WorkerJob | SyncWorkerJob:
        job = _UploadStreamJob(stream, mime_type, params=params, session=self, on_ready=on_ready,
                               callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    async def load_from(self, location: str, params: dict | None = None,
                        on_ready: Callable[[WorkerJob], None] | None = None) -> WorkerJob | SyncWorkerJob:
        job = _LoadFromJob(location=location, params=params, session=self, on_ready=on_ready,
                           callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    async def dev_mode_base_url(self) -> str:
        if self.worker_config is None:
            await self._reconnect()
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
        def noop():
            pass

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
        if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time < time.time() - FORCE_REFRESH_CONFIG_SECS:
            self.worker_config is None

        retried_re_auth = False
        retried_re_config = False
        start_time = time.time()
        while time.time() - start_time < MAX_RETRY_TIME_SECS:
            if self.worker_config is None:
                await self._reconnect()

            entry = self.load_balancer.next_entry(MAX_RETRY_TIME_SECS + 1)
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

            headers = {'Authorization': await self._authorization_header()}
            if accept is not None:
                headers['Accept'] = accept
            if content_type is not None:
                headers['Content-Type'] = content_type
            try:
                log_requests.debug('before %s %s', method, url)
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

                log_requests.debug('after %s %s status=%d headers=%s', method, url, response.status, response.headers)

                return response
            except aiohttp.ClientResponseError as e:
                if not retried_re_auth and e.status == 401:
                    # auth token might have just expired
                    log_requests.debug('after %s %s: 401, about to retry with fresh access token', method, url)
                    self.token = None
                    self.expire_token_time = None
                    retried_re_auth = True
                elif e.status == 404:
                    entry.mark_error()
                    log_requests.debug('after %s %s: 404, about to retry fail-over', method, url)
                else:
                    log_requests.exception('unexpected error', e)
                    raise e
            except aiohttp.ClientConnectionError:
                entry.mark_error()
                log_requests.debug('after %s %s: 404, about to retry fail-over', method, url)
            except Exception as e:
                log_requests.exception('unexpected error', e)
                raise e

        raise PopNotReachableException(self.pop_id, f"no success after {MAX_RETRY_TIME_SECS} secs")

    async def _worker_request_with_retry(self, method: str, url_path_and_query: str, accept: str | None = None,
                                         data: Any = None, content_type: str | None = None,
                                         timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        url = f'{await self.dev_mode_base_url()}/{url_path_and_query}'
        return await self.request_with_retry(method, url, accept, data, content_type, timeout)
