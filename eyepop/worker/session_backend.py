import asyncio
import json
import logging
import time
from io import StringIO
from typing import TYPE_CHECKING, Any, AsyncIterable, Callable
from urllib.parse import urlencode, urljoin

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.api import fetch_session_endpoint
from eyepop.compute.context import ComputeContext
from eyepop.exceptions import (
    ComputeSessionException,
    PopConfigurationException,
    PopNotReachableException,
    PopNotStartedException,
)
from eyepop.settings import settings
from eyepop.worker.load_balancer import EndpointLoadBalancer
from eyepop.worker.worker_backend import WorkerBackend
from eyepop.worker.worker_jobs import WorkerJob, WorkerSource, _iter_lines, _UploadSource
from eyepop.worker.worker_types import ComponentParams, Pop

if TYPE_CHECKING:
    from eyepop.worker.worker_endpoint import WorkerEndpoint

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')


class SessionBackend(WorkerBackend):
    """Runs jobs on a worker session: a compute API session, a named pop, or a worker found through its config."""

    def __init__(
            self,
            endpoint: 'WorkerEndpoint',
            pipeline_image: str | None,
            pipeline_version: str | None,
            session_name: str | None,
    ):
        self.endpoint = endpoint
        compute_ctx = endpoint.compute_ctx
        if compute_ctx:
            if pipeline_image:
                compute_ctx.pipeline_image = pipeline_image
            if pipeline_version:
                compute_ctx.pipeline_version = pipeline_version
            if session_name:
                compute_ctx.session_name = session_name
            compute_ctx.pop = endpoint.pop.model_dump() if endpoint.pop is not None else None
            self.is_dev_mode = not bool(endpoint.permanent_session_uuid)
        else:
            self.is_dev_mode = True

        self.worker_config = None
        self.load_balancer = EndpointLoadBalancer([])
        self._owns_pipeline_id = False
        self._pipeline_create_lock = asyncio.Lock()
        self.last_fetch_config_success_time = None
        self.last_fetch_config_error = None
        self.last_fetch_config_error_time = None

        endpoint.add_retry_handler(404, self._retry_404)

    @property
    def compute_ctx(self) -> ComputeContext | None:
        return self.endpoint.compute_ctx

    @compute_ctx.setter
    def compute_ctx(self, compute_ctx: ComputeContext | None):
        self.endpoint.compute_ctx = compute_ctx

    @property
    def client_session(self) -> aiohttp.ClientSession | None:
        return self.endpoint.client_session

    def _is_compute_transient(self) -> bool:
        return self.compute_ctx is not None and self.endpoint.permanent_session_uuid is None

    def _runs_on_instance_pipeline(self) -> bool:
        # An on-premise instance serves one fixed pop and refuses a pipeline of any other,
        # so a local endpoint without a pop of its own runs on the instance's pipeline.
        return self.endpoint.is_local_mode and self.compute_ctx is not None and self.compute_ctx.pop is None

    async def _authorization_headers(self) -> dict[str, str]:
        headers = {}
        authorization_header = await self.endpoint._authorization_header()
        if authorization_header is not None:
            headers['Authorization'] = authorization_header
        return headers

    async def _retry_404(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('after 404, about to retry with fresh config')
            self.worker_config = None
            self._owns_pipeline_id = False
            if self._is_compute_transient():
                assert self.compute_ctx is not None
                self.compute_ctx.pipeline_id = ""
                self.compute_ctx.pipeline_owned = False
            return True

    async def disconnect(self, timeout: float | None = None):
        client_timeout = None
        if timeout is not None:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
        if (self.is_dev_mode and self.endpoint.pop_id == 'transient'
                and self._has_pipeline_id()
                and (not self._is_compute_transient() or self._owns_pipeline_id)
                and self.client_session is not None):
            worker_config = self.worker_config
            assert worker_config is not None
            try:
                base_url = await self.dev_mode_base_url()
                delete_pipeline_url = f'{base_url}/pipelines/{worker_config["pipeline_id"]}'
                headers = await self._authorization_headers()
                await self.client_session.delete(delete_pipeline_url, headers=headers, timeout=client_timeout)
            except Exception as e:
                log.exception(e, exc_info=True)
            finally:
                worker_config.pop("pipeline_id", None)
                self._owns_pipeline_id = False
                if self.compute_ctx:
                    self.compute_ctx.pipeline_id = ""
                    self.compute_ctx.pipeline_owned = False

    async def connect(self):
        # Narrow Optional[ClientSession] — connect is only called after the endpoint's connect()
        assert self.client_session is not None
        if self.worker_config is not None:
            return

        if self.last_fetch_config_error_time is not None and self.last_fetch_config_error_time > time.time() - settings.min_config_reconnect_secs:
            if self.last_fetch_config_error is not None:
                raise self.last_fetch_config_error
            raise aiohttp.ClientConnectionError()

        if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time > time.time() - settings.min_config_reconnect_secs:
            raise aiohttp.ClientConnectionError()

        if self.compute_ctx:
            if not self.compute_ctx.session_endpoint:
                log_requests.debug("Fetching compute API session")
                self.compute_ctx = await fetch_session_endpoint(
                    compute_ctx=self.compute_ctx,
                    client_session=self.client_session,
                    permanent_session_uuid=self.endpoint.permanent_session_uuid,
                    is_local_mode=self.endpoint.is_local_mode,
                )
                self.endpoint.eyepop_url = self.compute_ctx.session_endpoint
                log_requests.debug(f"Compute session ready: {self.compute_ctx.session_endpoint}")

            pipeline_id = self.compute_ctx.pipeline_id
            self._owns_pipeline_id = bool(pipeline_id and self.compute_ctx.pipeline_owned)
            if self._is_compute_transient() and not self._owns_pipeline_id and not self._runs_on_instance_pipeline():
                pipeline_id = ""

            self.worker_config = {
                "session_endpoint": self.compute_ctx.session_endpoint,
                "pipeline_id": pipeline_id,
                "endpoints": [],
            }

            self.last_fetch_config_success_time = time.time()
            self.last_fetch_config_error = None
            self.last_fetch_config_error_time = None
        else:
            if self.endpoint.pop_id == 'transient':
                config_url = f'{self.endpoint.eyepop_url}/workers/config'
            else:
                config_url = f'{self.endpoint.eyepop_url}/pops/{self.endpoint.pop_id}/config?auto_start={self.endpoint.auto_start}'
            headers = await self._authorization_headers()
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
                    self.endpoint.token = None
                    self.endpoint.expire_token_time = None
                    headers = await self._authorization_headers()
                    async with self.client_session.get(config_url, headers=headers) as retried_response:
                        self.worker_config = await retried_response.json()
            except aiohttp.ClientConnectionError as e:
                self.last_fetch_config_error = e
                self.last_fetch_config_error_time = time.time()
                raise e

        if self.compute_ctx:
            log_requests.debug(f'Compute session: {self.compute_ctx.session_uuid}')

        if self.worker_config.get('status') == 'active_prod':
            self.is_dev_mode = False

        if (self.is_dev_mode and self.endpoint.stop_jobs and self._has_pipeline_id()
                and (not self._is_compute_transient() or self._owns_pipeline_id)):
            stop_jobs_url = f'{await self.dev_mode_pipeline_base_url()}/source?mode=preempt&processing=sync'
            body = {'sourceType': 'NONE'}
            headers = await self._authorization_headers()
            async with self.client_session.patch(stop_jobs_url, headers=headers, json=body) as response:
                pass

        if self.is_dev_mode and self.endpoint.pop is None and self._has_pipeline_id():
            get_url = await self.dev_mode_pipeline_base_url()
            headers = await self._authorization_headers()
            async with self.client_session.get(get_url, headers=headers) as response:
                response_json = await response.json()
                pop_as_dict = response_json.get('pop')
                # an on-premise instance reports its pipeline's pop without components
                if pop_as_dict is None or pop_as_dict.get('components') is None:
                    self.endpoint.pop = None
                else:
                    self.endpoint.pop = Pop(**pop_as_dict)
            log.debug('current pop is %s', self.endpoint.pop)

        self._configure_load_balancer()

    async def session(self) -> dict | None:
        session = await self.endpoint.access_session()
        if session is None:
            return None
        session['popId'] = self.endpoint.pop_id
        if self.worker_config:
            if 'session_endpoint' in self.worker_config:
                session['baseUrl'] = self.worker_config['session_endpoint']
            elif 'base_url' in self.worker_config:
                session['baseUrl'] = self.worker_config['base_url']
            session['pipelineId'] = self.worker_config.get('pipeline_id')
        return session

    async def set_pop(self, pop: Pop):
        if not self.is_dev_mode:
            raise PopConfigurationException(self.endpoint.pop_id, 'set_pop not supported in production mode')
        self.endpoint.pop = pop
        if self.compute_ctx:
            self.compute_ctx.pop = pop.model_dump()
        if self.client_session is None:
            return {"pop": pop.model_dump()}
        if self._is_compute_transient():
            return await self._set_pop_on_compute_transient(pop)
        if self.worker_config is None:
            await self.connect()
        if self.is_dev_mode and not self._has_pipeline_id():
            return await self._ensure_pipeline_started()
        response = await self.pipeline_patch('pop', content_type='application/json',
                                             data=pop.model_dump_json())
        return response

    async def _set_pop_on_compute_transient(self, pop: Pop):
        assert self.compute_ctx is not None
        assert self.client_session is not None

        self.compute_ctx.pop = pop.model_dump()
        self.compute_ctx.pipeline_id = ""
        self.compute_ctx.pipeline_owned = False
        self.compute_ctx = await fetch_session_endpoint(
            compute_ctx=self.compute_ctx,
            client_session=self.client_session,
            permanent_session_uuid=None,
        )
        self.endpoint.eyepop_url = self.compute_ctx.session_endpoint

        pipeline_id = self.compute_ctx.pipeline_id if self.compute_ctx.pipeline_owned else ""

        if self.worker_config is None:
            self.worker_config = {
                "session_endpoint": self.compute_ctx.session_endpoint,
                "pipeline_id": pipeline_id,
                "endpoints": [],
            }
        else:
            self.worker_config["session_endpoint"] = self.compute_ctx.session_endpoint
            self.worker_config["pipeline_id"] = pipeline_id

        self._owns_pipeline_id = bool(pipeline_id)
        if not self._has_pipeline_id():
            raise ComputeSessionException(
                "Compute transient session did not return an owned pipeline",
                session_uuid=self.compute_ctx.session_uuid,
            )

        self._configure_load_balancer()
        return {
            "session_uuid": self.compute_ctx.session_uuid,
            "session_endpoint": self.compute_ctx.session_endpoint,
            "pipeline_id": self.compute_ctx.pipeline_id,
        }

    async def dev_mode_pipeline_base_url(self) -> str:
        # Was `else: pass` which made return type str | None — raise explicitly instead
        if not self.is_dev_mode:
            raise PopNotStartedException(pop_id=self.endpoint.pop_id)
        base_url = await self.dev_mode_base_url()
        if self.worker_config is None:
            raise PopNotStartedException(pop_id=self.endpoint.pop_id)
        if not self._has_pipeline_id():
            await self._ensure_pipeline_started()
        return f'{base_url}/pipelines/{self.worker_config["pipeline_id"]}'

    async def dev_mode_base_url(self) -> str:
        if self.worker_config is None:
            await self.connect()
        if self.worker_config is None:
            raise PopNotStartedException(pop_id=self.endpoint.pop_id)
        if 'session_endpoint' in self.worker_config:
            return self.worker_config['session_endpoint'].rstrip("/")
        return urljoin(self.endpoint.eyepop_url, self.worker_config['base_url']).rstrip("/")

    def _has_pipeline_id(self) -> bool:
        return self.worker_config is not None and bool(self.worker_config.get('pipeline_id'))

    def _pipeline_create_body(self) -> dict[str, Any]:
        body = {
            "pop": self.endpoint.pop.model_dump() if self.endpoint.pop else {},
            "source": {
                "sourceType": "NONE"
            },
            "idleTimeoutSeconds": 300,
            "logging": ["out_meta"],
            "videoOutput": "no_output",
        }
        if self.endpoint.dataset_uuid is not None:
            body["datasetUuid"] = self.endpoint.dataset_uuid
        return body

    async def _ensure_pipeline_started(self) -> dict[str, Any] | None:
        if self._has_pipeline_id():
            return None
        async with self._pipeline_create_lock:
            if self._has_pipeline_id():
                return None
            return await self._create_pipeline()

    async def _create_pipeline(self) -> dict[str, Any]:
        assert self.client_session is not None
        if self.worker_config is None:
            await self.connect()
        if self.worker_config is None:
            raise PopNotStartedException(pop_id=self.endpoint.pop_id)

        start_pipeline_url = f'{await self.dev_mode_base_url()}/pipelines'
        headers = await self._authorization_headers()

        async with self.client_session.post(
                start_pipeline_url,
                headers=headers,
                json=self._pipeline_create_body(),
        ) as response:
            response.raise_for_status()
            response_json = await response.json()

        self.worker_config['pipeline_id'] = response_json['id']
        self._owns_pipeline_id = True
        if self.compute_ctx:
            self.compute_ctx.pipeline_uuid = response_json['id']
            self.compute_ctx.pipeline_id = response_json['id']
            self.compute_ctx.pipeline_owned = True
            log.debug(f"Created pipeline with ID: {response_json['id']}")

        self._configure_load_balancer()
        return response_json

    def _configure_load_balancer(self):
        if self.worker_config is None:
            return

        if self.is_dev_mode:
            if not self._has_pipeline_id():
                self.load_balancer = EndpointLoadBalancer([])
                return
            if 'session_endpoint' in self.worker_config:
                base_url = self.worker_config['session_endpoint'].rstrip("/")
            else:
                base_url = urljoin(self.endpoint.eyepop_url, self.worker_config['base_url']).rstrip("/")
            endpoint = {'base_url': base_url, 'pipeline_id': self.worker_config['pipeline_id']}
            self.load_balancer = EndpointLoadBalancer([endpoint])
            log.debug(f"Initialized load balancer with endpoint: {endpoint}")
        else:
            if 'session_endpoint' in self.worker_config:
                base_url = self.worker_config['session_endpoint'].rstrip("/")
                endpoint = {'base_url': base_url, 'pipeline_id': self.worker_config['pipeline_id']}
                self.load_balancer = EndpointLoadBalancer([endpoint])
                log.debug(f"Initialized load balancer with endpoint: {endpoint}")
            else:
                self.load_balancer = EndpointLoadBalancer(self.worker_config['endpoints'])
                log.debug(f"Initialized load balancer with endpoints: {self.worker_config['endpoints']}")

    #
    # The worker session's pipeline routes.
    # Return types changed from _RequestContextManager to ClientResponse — these methods
    # await the request and return the response directly, not a context manager.
    #

    async def pipeline_get(self, url_path_and_query: str, accept: str | None = None,
                           timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._pipeline_request_with_retry('GET', url_path_and_query, accept=accept, timeout=timeout)

    async def pipeline_post(self, url_path_and_query: str, accept: str | None = None, open_data: Callable | None = None,
                            content_type: str | None = None,
                            timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._pipeline_request_with_retry('POST', url_path_and_query, accept=accept, open_data=open_data,
                                                       content_type=content_type, timeout=timeout)

    async def pipeline_patch(self, url_path_and_query: str, accept: str | None = None, data: Any = None,
                             content_type: str | None = None,
                             timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        def open_data():
            if isinstance(data, str):
                return StringIO(data)
            else:
                return data

        return await self._pipeline_request_with_retry('PATCH', url_path_and_query, accept=accept, open_data=open_data,
                                                       content_type=content_type, timeout=timeout)

    async def pipeline_delete(self, url_path_and_query: str,
                              timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._pipeline_request_with_retry('DELETE', url_path_and_query, timeout=timeout)

    async def _pipeline_request_with_retry(self, method: str, url_path_and_query: str, accept: str | None = None,
                                           open_data: Callable | None = None, content_type: str | None = None,
                                           timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        # Narrow Optional[ClientSession] for all .request() calls below
        assert self.client_session is not None
        if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time < time.time() - settings.force_refresh_config_secs:
            self.worker_config = None

        failed_attempts = 0
        retried_re_config = False
        start_time = time.time()

        while time.time() - start_time < settings.max_retry_time_secs:
            if self.worker_config is None:
                await self.connect()
            if self.is_dev_mode and not self._has_pipeline_id():
                await self._ensure_pipeline_started()

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
                    raise PopNotReachableException(self.endpoint.pop_id, self.load_balancer.get_debug_status())

            url = f'{entry.base_url}/pipelines/{entry.pipeline_id}/{url_path_and_query}'

            headers = await self._authorization_headers()
            if accept is not None:
                headers['Accept'] = accept
            if content_type is not None:
                headers['Content-Type'] = content_type
            try:
                if open_data is not None:
                    data = open_data()
                    if isinstance(data, StringIO):
                        response = await self.client_session.request(method, url, headers=headers,
                                                                     data=data.getvalue(), timeout=timeout)
                    elif isinstance(data, AsyncIterable):
                        async def data_streamer(_data: AsyncIterable):
                            async for chunk in _data:
                                yield chunk
                        response = await self.client_session.request(method, url, headers=headers, data=data_streamer(data),
                                                                     timeout=timeout)
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
                    if e.status not in self.endpoint.retry_handlers:
                        log_requests.exception('unexpected error: %s', e)
                        raise e
                    if not await self.endpoint.retry_handlers[e.status](e.status, failed_attempts):
                        log_requests.exception('unexpected error')
                        raise e
            except aiohttp.ClientConnectionError:
                entry.mark_error()
            except Exception as e:
                log_requests.exception('unexpected error')
                raise e

        raise PopNotReachableException(self.endpoint.pop_id, [])

    #
    # Jobs
    #

    async def run(self, job: WorkerJob):
        source = job.source
        if source.uploads:
            await self._run_upload(job, source)
        elif source.asset_uuid is not None:
            await self._run_patch(job, _asset_body(source))
        else:
            await self._run_patch(job, _url_body(source))

    async def _run_patch(self, job: WorkerJob, body: dict[str, Any]):
        job.set_response(await self.pipeline_patch('source?mode=queue&processing=sync',
                                                   accept='application/jsonl',
                                                   data=json.dumps(body),
                                                   content_type='application/json',
                                                   timeout=aiohttp.ClientTimeout(total=None, sock_read=600)))
        await _read_response(job)

    async def _run_upload(self, job: WorkerJob, source: WorkerSource):
        """Uploads one or more media items as a single source.

        A single item is posted as a raw body (or multipart when
        params/roi/fps are present, or full-duplex for video). Two or more items are
        posted together as one multipart request with an ordered 'file' part each -
        an image group (server SOURCE_GROUP): one inference unit, one prediction
        stream, member order preserved.
        """
        uploads = source.uploads
        query_params: dict[str, Any] = {
            "mode": "queue",
        }
        if source.video_mode is not None:
            query_params['videoMode'] = source.video_mode.value
        if source.is_live is not None:
            query_params["isLive"] = source.is_live
        if source.version is not None:
            query_params['version'] = source.version
        if source.motion_detect is not None:
            query_params.update(source.motion_detect.model_dump(exclude_none=True))
        if source.media_cache_seconds is not None:
            query_params['mediaCacheSeconds'] = source.media_cache_seconds

        # A single item with no extra parts can stream its raw body; anything
        # else (extra parts, or a multi-item image group) is sent as multipart.
        single_source = uploads[0] if len(uploads) == 1 else None
        no_extra_parts = (source.component_params is None and source.roi is None
                          and source.fps is None and source.camera is None)
        # Full duplex is only used for a single video upload; an image group is
        # always posted as one sync multipart request.
        needs_full_duplex = (
            single_source is not None
            and single_source.mime_type is not None
            and single_source.mime_type.startswith("video/")
        )
        timeout = aiohttp.ClientTimeout(total=None, sock_read=600)

        def open_mp_writer():
            return _multipart(source)

        if needs_full_duplex:
            response = await self.pipeline_post('prepareSource?timeout=600s', accept='application/jsonl',
                                                timeout=timeout)
            job.set_response(response)
            # Read first event to get the prepared source id and then use
            # two HTTP connections to simulate ful duplex HTTP.
            source_id = None
            line = await response.content.readline()
            if line:
                json_response = json.loads(line)
                event = json_response.get('event', None)
                if event is not None:
                    if event.get('type', None) == 'prepared':
                        source_id = event.get('source_id', None)
            if source_id is None:
                raise ValueError("did not get a prepared sourceId to simulate full duplex")
            query_params['processing'] = 'async'
            query_params['sourceId'] = source_id
            upload_url = f'source?{urlencode(query_params)}'
            if single_source is not None and no_extra_parts:
                upload_coro = self.pipeline_post(upload_url, accept='application/jsonl',
                                                 open_data=single_source.open_stream,
                                                 content_type=single_source.mime_type, timeout=timeout)
            else:
                upload_coro = self.pipeline_post(upload_url, accept='application/jsonl',
                                                 open_data=open_mp_writer, timeout=timeout)
            await asyncio.gather(upload_coro, _read_response(job))
        else:
            query_params['processing'] = 'sync'
            upload_url = f'source?{urlencode(query_params)}'
            if single_source is not None and no_extra_parts:
                job.set_response(await self.pipeline_post(upload_url, accept='application/jsonl',
                                                          open_data=single_source.open_stream,
                                                          content_type=single_source.mime_type, timeout=timeout))
            else:
                job.set_response(await self.pipeline_post(upload_url, accept='application/jsonl',
                                                          open_data=open_mp_writer, timeout=timeout))
            await _read_response(job)


async def _read_response(job: WorkerJob) -> bool:
    got_results = False
    response = job._response
    if response is not None:
        try:
            job.callback.first_result(job)
            async for line in _iter_lines(response.content):
                # TODO aiohttp should do do this internally
                for trace in response._traces:
                    await trace.send_response_chunk_received(
                        response.method, response.url, line
                    )
                if not got_results:
                    got_results = True
                prediction = json.loads(line)
                await job.push_message(prediction)
        finally:
            response.close()
    return got_results


def _multipart(source: WorkerSource) -> aiohttp.MultipartWriter:
    mp_writer = aiohttp.MultipartWriter('form-data')
    if source.component_params is not None:
        component_params_part = mp_writer.append_json(
            TypeAdapter(list[ComponentParams]).dump_python(source.component_params))
        component_params_part.set_content_disposition(
            'form-data', name='params', filename='blob')
    if source.roi is not None:
        roi_part = mp_writer.append_json(source.roi.model_dump(exclude_none=True))
        roi_part.set_content_disposition('form-data', name='roi', filename='blob')
    if source.fps is not None:
        fps_part = mp_writer.append_json(source.fps)
        fps_part.set_content_disposition('form-data', name='fps', filename='blob')
    if source.camera is not None:
        camera_part = mp_writer.append_json(source.camera.model_dump(exclude_none=True))
        camera_part.set_content_disposition('form-data', name='camera', filename='blob')

    for upload in source.uploads:
        file_part = _file_part(mp_writer, upload)
        file_part.set_content_disposition('form-data', name='file', filename='blob')
    return mp_writer


def _file_part(mp_writer: aiohttp.MultipartWriter, upload: _UploadSource):
    if upload.mime_type is not None:
        return mp_writer.append(upload.open_stream(), {'Content-Type': upload.mime_type})
    return mp_writer.append(upload.open_stream())


def _url_body(source: WorkerSource) -> dict[str, Any]:
    """A single URL is a normal `URL` source; two or more form one `GROUP` source, member order preserved."""
    if not source.urls:
        raise ValueError("load_from requires at least one url")
    if len(source.urls) == 1:
        # Single URL: unchanged body (field order preserved for compatibility).
        body: dict[str, Any] = {
            "sourceType": "URL",
            "url": source.urls[0],
            "version": source.version,
        }
        if source.motion_detect is not None:
            body.update(source.motion_detect.model_dump(exclude_none=True))
        if source.component_params is not None:
            body['params'] = TypeAdapter(list[ComponentParams]).dump_python(source.component_params)
        if source.roi is not None:
            body['roi'] = source.roi.model_dump(exclude_none=True)
        if source.fps is not None:
            body['fps'] = source.fps
        if source.camera is not None:
            body['camera'] = source.camera.model_dump(exclude_none=True)
        if source.media_cache_seconds is not None:
            body['mediaCacheSeconds'] = source.media_cache_seconds
        if source.rtsp_force_non_compliant_url is not None:
            body['rtspForceNonCompliantUrl'] = source.rtsp_force_non_compliant_url
        return body
    # Two or more URLs: one image group (no video/motion/fps).
    body = {
        "sourceType": "GROUP",
        "sources": [{"sourceType": "URL", "url": location} for location in source.urls],
        "version": source.version,
    }
    if source.component_params is not None:
        body['params'] = TypeAdapter(list[ComponentParams]).dump_python(source.component_params)
    if source.roi is not None:
        body['roi'] = source.roi.model_dump(exclude_none=True)
    if source.camera is not None:
        body['camera'] = source.camera.model_dump(exclude_none=True)
    if source.media_cache_seconds is not None:
        body['mediaCacheSeconds'] = source.media_cache_seconds
    return body


def _asset_body(source: WorkerSource) -> dict[str, Any]:
    body: dict[str, Any] = {
        "sourceType": "ASSET_UUID",
        "assetUuid": source.asset_uuid,
        "version": source.version
    }
    if source.motion_detect is not None:
        body.update(source.motion_detect.model_dump())
    if source.roi is not None:
        body['roi'] = source.roi.model_dump(exclude_none=True)
    if source.component_params is not None:
        body['params'] = TypeAdapter(list[ComponentParams]).dump_python(source.component_params)
    if source.camera is not None:
        body['camera'] = source.camera.model_dump(exclude_none=True)
    if source.media_cache_seconds is not None:
        body['mediaCacheSeconds'] = source.media_cache_seconds
    return body
