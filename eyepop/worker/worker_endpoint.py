import json
import logging
import os
import time
import warnings
from io import StringIO
from typing import TYPE_CHECKING, Any, BinaryIO, Callable
from urllib.parse import urljoin

import aiohttp
from deprecated import deprecated
from eyepop.compute import fetch_session_endpoint, ComputeContext
from eyepop.endpoint import Endpoint
from eyepop.exceptions import (
    PopConfigurationException,
    PopNotReachableException,
    PopNotStartedException,
)
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

if TYPE_CHECKING:  # noqa: F821
    from aiohttp import _RequestContextManager

log = logging.getLogger("eyepop")
log_requests = logging.getLogger("eyepop.requests")
log_metrics = logging.getLogger("eyepop.metrics")

MIN_CONFIG_RECONNECT_SECS = 10.0
MAX_RETRY_TIME_SECS = 30.0
FORCE_REFRESH_CONFIG_SECS = 61.0 * 61.0


def _warn_deprecated(message: str):
    warnings.warn(message, DeprecationWarning)  # noqa: B028


class WorkerEndpoint(Endpoint, WorkerClientSession):
    """Endpoint to an EyePop.ai worker."""

    def __init__(
            self,
            secret_key: str | None,
            access_token: str | None,
            eyepop_url: str,
            pop_id: str,
            auto_start: bool,
            stop_jobs: bool,
            job_queue_length: int,
            is_sandbox: bool,
            request_tracer_max_buffer: int,
            dataset_uuid: str | None = None,
    ):
        super().__init__(
            secret_key=secret_key,
            access_token=access_token,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer
        )
        self.pop_id = pop_id
        self.auto_start = auto_start
        self.stop_jobs = stop_jobs
        self.is_sandbox = is_sandbox
        self.dataset_uuid = dataset_uuid

        self.sandbox_id = None

        self.is_dev_mode = True

        self.worker_config = None
        self.last_fetch_config_success_time = None
        self.last_fetch_config_error = None
        self.last_fetch_config_error_time = None

        # old way, TODO remove
        self.pop_comp = None
        self.model_refs = []
        self.post_transform = None

        # new way
        self.pop = None

        self.add_retry_handler(404, self._retry_404)

    async def _retry_404(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug("after 404, about to retry with fresh config")
            self.worker_config = None
            return True

    async def _disconnect(self, timeout: float | None = None):
        client_timeout = None
        if timeout is not None:
            client_timeout = aiohttp.ClientTimeout(total=timeout)
        if (
            self.is_dev_mode
            and self.pop_id == "transient"
            and self.worker_config is not None
            and self.worker_config["pipeline_id"] is not None
        ):
            try:
                base_url = await self.dev_mode_base_url()
                delete_pipeline_url = f"{base_url}/pipelines/{self.worker_config['pipeline_id']}"
                headers = {}
                authorization_header = await self._authorization_header()
                if authorization_header is not None:
                    headers["Authorization"] = authorization_header
                log_requests.debug("before DELETE %s", delete_pipeline_url)
                await self.client_session.delete(
                    delete_pipeline_url, headers=headers, timeout=client_timeout
                )
                log_requests.debug("after DELETE %s", delete_pipeline_url)
            except Exception as e:
                log.exception(e, exc_info=True)
            finally:
                del self.worker_config["pipeline_id"]

        if self.sandbox_id is not None:
            try:
                base_url = await self.dev_mode_base_url()
                delete_sandbox_url = f"{base_url}/sandboxes/{self.sandbox_id}"
                headers = {}
                authorization_header = await self._authorization_header()
                if authorization_header is not None:
                    headers["Authorization"] = authorization_header
                log_requests.debug("before DELETE %s", delete_sandbox_url)
                await self.client_session.delete(
                    delete_sandbox_url, headers=headers, timeout=client_timeout
                )
                log_requests.debug("after DELETE %s", delete_sandbox_url)
            except Exception as e:
                log.exception(e, exc_info=True)
            finally:
                self.sandbox_id = None

    async def _reconnect(self):
        if self.worker_config is not None:
            return

        if (
            self.last_fetch_config_error_time is not None
            and self.last_fetch_config_error_time > time.time() - MIN_CONFIG_RECONNECT_SECS
        ):
            raise self.last_fetch_config_error

        if (
            self.last_fetch_config_success_time is not None
            and self.last_fetch_config_success_time > time.time() - MIN_CONFIG_RECONNECT_SECS
        ):
            raise aiohttp.ClientConnectionError()

        # Always use compute API - EYEPOP_URL should be the compute API URL
        try:
            compute_config = ComputeContext(
                compute_url=self.eyepop_url,  # Use eyepop_url as compute API URL
                user_uuid=os.getenv("EYEPOP_USER_UUID", ""),
                secret_key=self.secret_key,  # Use the secret key from endpoint
                wait_for_session_timeout=30,
                wait_for_session_interval=2,
            )
            compute_context = fetch_session_endpoint(compute_config)
            if compute_context.session_endpoint:
                # Store data API URL for the session
                self.data_api_url = os.getenv("EYEPOP_DATA_API", "https://dataset-api.staging.eyepop.xyz")
                
                self.worker_config = {
                    "base_url": compute_context.session_endpoint,
                    "pipeline_id": compute_context.pipeline_uuid,
                    "status": "active_prod",
                }
                self.last_fetch_config_success_time = time.time()
                self.last_fetch_config_error = None
                self.last_fetch_config_error_time = None
                # With compute API, we're always in "dev mode" for dynamic configuration
                self.is_dev_mode = True
                # Store compute API token for authorization
                self.compute_api_token = self.secret_key
                # For transient mode, we need to create the pipeline
                self.pipeline_id = None  # Will be created on first use
                endpoint = {"base_url": compute_context.session_endpoint, "pipeline_id": None}
                self.load_balancer = EndpointLoadBalancer([endpoint])
                return
        except Exception as e:
            self.last_fetch_config_error = e
            self.last_fetch_config_error_time = time.time()
            log_requests.error(f"Failed to fetch session from compute API: {e}")
            raise aiohttp.ClientConnectionError(f"Failed to fetch session from compute API: {e}")

        # Remove web-api fallback - only use compute API
        # This code path should not be reached anymore
        raise RuntimeError("Only compute API is supported. Ensure EYEPOP_URL points to compute API.")
        # All web-api code removed - only compute API is supported
        # All exception handling for web-api removed

        # Dev mode and sandbox handling removed - compute API handles all modes
        # The following code is legacy and will be removed
        return

        if False and self.is_sandbox and self.sandbox_id is None:
            create_sandbox_url = f"{base_url}/sandboxes"
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers["Authorization"] = authorization_header
            log_requests.debug("before POST %s", create_sandbox_url)
            async with self.client_session.post(create_sandbox_url, headers=headers) as response:
                response_json = await response.json()
            log_requests.debug("after POST %s", create_sandbox_url)
            self.sandbox_id = response_json

        if self.pop_id == "transient":
            if self.sandbox_id is None:
                start_pipeline_url = f"{base_url}/pipelines"
            else:
                start_pipeline_url = f"{base_url}/pipelines?sandboxId={self.sandbox_id}"
            if self.pop is not None:
                body = {
                    "pop": self.pop,
                    "source": {
                        "sourceType": "NONE"
                    },
                    "idleTimeoutSeconds": 60,
                    "logging": ["out_meta"],
                    "videoOutput": "no_output",
                }
                if self.dataset_uuid is not None:
                    body["datasetUuid"] = self.dataset_uuid
            else:
                if self.pop_comp is None:
                    self.pop_comp = 'identity'
                body = {
                    'inferPipelineDef': {
                        'pipeline': self.pop_comp,
                        'modelRefs': self.model_refs
                    },
                    'postTransformDef': {
                        'transform': self.post_transform
                    },
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
                headers["Authorization"] = authorization_header
            log_requests.debug("before POST %s", start_pipeline_url)
            async with self.client_session.post(
                start_pipeline_url, headers=headers, json=body
            ) as response:
                response_json = await response.json()
            log_requests.debug("after POST %s", start_pipeline_url)
            self.worker_config["pipeline_id"] = response_json["id"]

        if self.is_dev_mode and (
            self.worker_config["base_url"] is None or self.worker_config["pipeline_id"] is None
        ):
            raise PopNotStartedException(pop_id=self.pop_id)

        if self.is_dev_mode and self.stop_jobs:
            stop_jobs_url = (
                f"{await self.dev_mode_pipeline_base_url()}/source?mode=preempt&processing=sync"
            )
            body = {"sourceType": "NONE"}
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers["Authorization"] = authorization_header
            log_requests.debug("before PATCH %s", stop_jobs_url)
            async with self.client_session.patch(
                stop_jobs_url, headers=headers, json=body
            ) as response:
                pass
            log_requests.debug("after PATCH %s", stop_jobs_url)

        if self.is_dev_mode and self.pop_comp is None:
            # get current pipeline string and store
            get_url = await self.dev_mode_pipeline_base_url()
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers["Authorization"] = authorization_header
            log_requests.debug("before GET %s", get_url)
            async with self.client_session.get(get_url, headers=headers) as response:
                response_json = await response.json()
                self.pop_comp = response_json.get("inferPipeline")
                self.post_transform = response_json.get("postTransform")
            log_requests.debug("after GET %s", get_url)
            log.debug("current popComp is %s", self.pop_comp)

        if self.is_dev_mode:
            endpoint = {
                "base_url": urljoin(self.eyepop_url, self.worker_config["base_url"]).rstrip("/"),
                "pipeline_id": self.worker_config["pipeline_id"],
            }
            self.load_balancer = EndpointLoadBalancer([endpoint])
        else:
            self.load_balancer = EndpointLoadBalancer(self.worker_config["endpoints"])

    async def session(self) -> dict:
        session = await super().session()
        session["popId"] = self.pop_id
        if self.worker_config:
            session["baseUrl"] = self.worker_config["base_url"]
            session["pipelineId"] = self.worker_config["pipeline_id"]
        session["sandboxId"] = self.sandbox_id
        return session

    async def get_pop(self) -> Pop | None:
        return self.pop

    async def set_pop(self, pop: Pop):
        if not self.is_dev_mode:
            raise PopConfigurationException(
                self.pop_id, "set_pop_comp not supported in production mode"
            )
        
        # Store the pop for later use
        self.pop = pop
        self.pop_comp = None
        self.model_refs = []
        self.post_transform = None
        
        # For compute API with transient mode, create pipeline on first use
        if hasattr(self, 'compute_api_token'):
            if not hasattr(self, 'pipeline_id') or self.pipeline_id is None:
                await self._create_compute_pipeline()
                return {"status": "pipeline_created"}
            else:
                # Pipeline already exists, just update it
                response = await self.pipeline_patch(
                    "pop", content_type="application/json", data=pop.model_dump_json()
                )
                return response
        
        # Legacy path for non-compute API
        response = await self.pipeline_patch(
            "pop", content_type="application/json", data=pop.model_dump_json()
        )
        return response

    @deprecated(version="1.0.0", reason="Use get_pop() instead")
    async def get_pop_comp(self) -> str:
        return self.pop_comp

    @deprecated(version="1.0.0", reason="Use set_pop() instead")
    async def set_pop_comp(self, pop_comp: str = None, model_refs: list[dict] = []):  # noqa: B006
        if not self.is_dev_mode:
            raise PopConfigurationException(
                self.pop_id, "set_pop_comp not supported in production mode"
            )
        response = await self.pipeline_patch(
            "inferencePipeline",
            content_type="application/json",
            data=json.dumps({"pipeline": pop_comp, "modelRefs": model_refs}),
        )
        self.pop_comp = pop_comp
        self.model_refs = model_refs
        return response

    @deprecated(version="1.0.0", reason="Use get_pop() instead")
    async def get_post_transform(self) -> str:
        return self.post_transform

    @deprecated(version="1.0.0", reason="Use set_pop() instead")
    async def set_post_transform(self, transform: str = None):
        if not self.is_dev_mode:
            raise PopConfigurationException(
                self.pop_id, "set_post_transform not supported in production mode"
            )
        response = await self.pipeline_patch(
            "postTransform",
            content_type="application/json",
            data=json.dumps({"transform": transform}),
        )
        self.post_transform = transform
        return response

    async def list_models(self) -> list[dict]:
        _warn_deprecated("list_models for development use only")
        if self.sandbox_id is None:
            get_path = "models/instances"
        else:
            get_path = f"models/instances?sandboxId={self.sandbox_id}"

        response = await self._worker_request_with_retry("GET", url_path_and_query=get_path)
        return await response.json()

    async def get_manifest(self) -> list[dict]:
        _warn_deprecated("get_manifest for development use only")
        if self.sandbox_id is None:
            get_path = "models/sources"
        else:
            get_path = f"models/sources?sandboxId={self.sandbox_id}"
        response = await self._worker_request_with_retry("GET", url_path_and_query=get_path)
        return await response.json()

    async def set_manifest(self, manifests: list[dict]):
        if not self.is_dev_mode:
            raise PopConfigurationException(
                self.pop_id, "set_manifest not supported in production mode"
            )
        _warn_deprecated("set_manifest for development use only")
        if self.sandbox_id is None:
            put_path = "models/sources"
        else:
            put_path = f"models/sources?sandboxId={self.sandbox_id}"
        await self._worker_request_with_retry(
            "PUT",
            url_path_and_query=put_path,
            content_type="application/json",
            data=json.dumps(manifests),
        )

    async def load_model(self, model: dict, override: bool = False) -> dict:
        if not self.is_dev_mode:
            raise PopConfigurationException(
                self.pop_id, "load_model not supported in production mode"
            )
        _warn_deprecated("load_model for development use only")
        if override:
            await self.unload_model(model)
        if self.sandbox_id is None:
            post_path = "models/instances"
        else:
            post_path = f"models/instances?sandboxId={self.sandbox_id}"
        response = await self._worker_request_with_retry(
            "POST",
            url_path_and_query=post_path,
            content_type="application/json",
            data=json.dumps(model),
        )
        return await response.json()

    async def unload_model(self, model_id: str):
        if not self.is_dev_mode:
            raise PopConfigurationException(
                self.pop_id, "unload_model not supported in production mode"
            )
        _warn_deprecated("purge_model for development use only")
        if self.sandbox_id is None:
            delete_path = 'models/instances/{model["id"]}'
        else:
            delete_path = f"models/instances/{model_id}?sandboxId={self.sandbox_id}"

        await self._worker_request_with_retry("DELETE", url_path_and_query=delete_path)

    async def dev_mode_pipeline_base_url(self):
        if self.is_dev_mode:
            return f"{await self.dev_mode_base_url()}/pipelines/{self.worker_config['pipeline_id']}"
        else:
            pass

    async def upload(
        self,
        location: str,
        video_mode: VideoMode | None = None,
        params: list[ComponentParams] | None = None,
        on_ready: Callable[[WorkerJob], None] | None = None,
    ) -> WorkerJob | SyncWorkerJob:
        job = _UploadFileJob(
            location=location,
            video_mode=video_mode,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector,
        )
        await self._task_start(job.execute())
        return job

    async def upload_stream(
        self,
        stream: BinaryIO,
        mime_type: str,
        video_mode: VideoMode | None = None,
        params: list[ComponentParams] | None = None,
        on_ready: Callable[[WorkerJob], None] | None = None,
    ) -> WorkerJob | SyncWorkerJob:
        job = _UploadStreamJob(
            stream=stream,
            mime_type=mime_type,
            video_mode=video_mode,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector,
        )
        await self._task_start(job.execute())
        return job

    async def load_from(
        self,
        location: str,
        params: list[ComponentParams] | None = None,
        on_ready: Callable[[WorkerJob], None] | None = None,
    ) -> WorkerJob | SyncWorkerJob:
        job = _LoadFromJob(
            location=location,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector,
        )
        await self._task_start(job.execute())
        return job

    async def load_asset(
        self,
        asset_uuid: str,
        params: list[ComponentParams] | None = None,
        on_ready: Callable[[WorkerJob], None] | None = None,
    ) -> WorkerJob | SyncWorkerJob:
        job = _LoadFromAssetUuidJob(
            asset_uuid=asset_uuid,
            component_params=params,
            session=self,
            on_ready=on_ready,
            callback=self.metrics_collector,
        )
        await self._task_start(job.execute())
        return job

    async def dev_mode_base_url(self) -> str:
        if self.worker_config is None:
            await self._reconnect()
        return urljoin(self.eyepop_url, self.worker_config["base_url"]).rstrip("/")
    
    async def _create_compute_pipeline(self):
        """Create a pipeline on the compute API session endpoint"""
        if not hasattr(self, 'compute_api_token'):
            raise RuntimeError("Not using compute API")
        
        base_url = self.worker_config["base_url"]
        
        # Create an empty pipeline first (avoids model alias resolution issues)
        body = {
            "pop": {"components": []},  # Start with empty components
            "source": {"sourceType": "NONE"},
            "idleTimeoutSeconds": 60,
            "logging": ["out_meta"],
            "videoOutput": "no_output"
        }
        
        if self.dataset_uuid is not None:
            body["datasetUuid"] = self.dataset_uuid
        
        headers = {}
        authorization_header = await self._authorization_header()
        if authorization_header is not None:
            headers["Authorization"] = authorization_header
        
        create_url = f"{base_url}/pipelines"
        log_requests.debug("Creating pipeline at %s", create_url)
        
        async with self.client_session.post(create_url, headers=headers, json=body) as response:
            response_json = await response.json()
            self.pipeline_id = response_json["id"]
            self.worker_config["pipeline_id"] = self.pipeline_id
            
            # Update load balancer with the new pipeline ID
            endpoint = {"base_url": base_url, "pipeline_id": self.pipeline_id}
            self.load_balancer = EndpointLoadBalancer([endpoint])
            
            log_requests.debug("Pipeline created with ID: %s", self.pipeline_id)
            
            # Now update the pipeline with the actual pop configuration
            if self.pop:
                await self.pipeline_patch(
                    "pop", content_type="application/json", data=self.pop.model_dump_json()
                )
            
            return self.pipeline_id

    #
    # Implements: _WorkerClientSession
    #

    async def pipeline_get(
        self,
        url_path_and_query: str,
        accept: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> "_RequestContextManager":
        return await self._pipeline_request_with_retry(
            "GET", url_path_and_query, accept=accept, timeout=timeout
        )

    async def pipeline_post(
        self,
        url_path_and_query: str,
        accept: str | None = None,
        open_data: Callable = None,
        content_type: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> "_RequestContextManager":
        return await self._pipeline_request_with_retry(
            "POST",
            url_path_and_query,
            accept=accept,
            open_data=open_data,
            content_type=content_type,
            timeout=timeout,
        )

    async def pipeline_patch(
        self,
        url_path_and_query: str,
        accept: str | None = None,
        data: Any = None,
        content_type: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> "_RequestContextManager":
        def noop():
            pass

        def open_data():
            if isinstance(data, str):
                return StringIO(data)
            else:
                return data

        return await self._pipeline_request_with_retry(
            "PATCH",
            url_path_and_query,
            accept=accept,
            open_data=open_data,
            content_type=content_type,
            timeout=timeout,
        )

    async def pipeline_delete(
        self, url_path_and_query: str, timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        return await self._pipeline_request_with_retry(
            "DELETE", url_path_and_query, timeout=timeout
        )

    async def _pipeline_request_with_retry(
        self,
        method: str,
        url_path_and_query: str,
        accept: str | None = None,
        open_data: Callable = None,
        content_type: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> "_RequestContextManager":
        # @torsten: what is this meant to do? assign or actually check something
        # if self.last_fetch_config_success_time is not None and self.last_fetch_config_success_time < time.time() - FORCE_REFRESH_CONFIG_SECS:
        #     self.worker_config is None

        failed_attempts = 0
        _retried_re_auth = False
        retried_re_config = False
        start_time = time.time()
        while time.time() - start_time < MAX_RETRY_TIME_SECS:
            if self.worker_config is None:
                await self._reconnect()

            entry = self.load_balancer.next_entry(MAX_RETRY_TIME_SECS + 1)
            if entry is None:
                if not retried_re_config:
                    # pipeline might have just shut down
                    log_requests.debug("no healthy endpoints, about to retry with fresh config")
                    self.worker_config = None
                    retried_re_config = True
                    continue
                else:
                    raise PopNotReachableException(
                        self.pop_id, self.load_balancer.get_debug_status()
                    )

            url = f"{entry.base_url}/pipelines/{entry.pipeline_id}/{url_path_and_query}"

            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers["Authorization"] = authorization_header
            if accept is not None:
                headers["Accept"] = accept
            if content_type is not None:
                headers["Content-Type"] = content_type
            try:
                log_requests.debug("before %s %s", method, url)
                if open_data is not None:
                    with open_data() as data:
                        if isinstance(data, StringIO):
                            response = await self.client_session.request(
                                method, url, headers=headers, data=data.getvalue(), timeout=timeout
                            )
                        else:
                            response = await self.client_session.request(
                                method, url, headers=headers, data=data, timeout=timeout
                            )
                else:
                    response = await self.client_session.request(
                        method, url, headers=headers, timeout=timeout
                    )

                entry.mark_success()

                log_requests.debug(
                    "after %s %s status=%d headers=%s",
                    method,
                    url,
                    response.status,
                    response.headers,
                )

                return response
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    # in load balanced configuration, we overwrite the standard 404 handler
                    entry.mark_error()
                    log_requests.debug("after %s %s: 404, about to retry fail-over", method, url)
                else:
                    failed_attempts += 1
                    if e.status not in self.retry_handlers:
                        log_requests.exception("unexpected error", e)
                        raise e
                    if not await self.retry_handlers[e.status](e.status, failed_attempts):
                        log_requests.exception("unexpected error", e)
                        raise e
            except aiohttp.ClientConnectionError:
                entry.mark_error()
                log_requests.debug("after %s %s: 404, about to retry fail-over", method, url)
            except Exception as e:
                log_requests.exception("unexpected error", e)
                raise e

        raise PopNotReachableException(self.pop_id, f"no success after {MAX_RETRY_TIME_SECS} secs")

    async def _worker_request_with_retry(
        self,
        method: str,
        url_path_and_query: str,
        accept: str | None = None,
        data: Any = None,
        content_type: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> "_RequestContextManager":
        url = f"{await self.dev_mode_base_url()}/{url_path_and_query}"
        return await self.request_with_retry(method, url, accept, data, content_type, timeout)
