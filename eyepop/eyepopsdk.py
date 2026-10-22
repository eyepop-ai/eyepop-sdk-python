import logging
import os

from typing_extensions import deprecated

from eyepop import __version__
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_syncify import SyncDataEndpoint
from eyepop.settings import account_uuid_from_env, settings
from eyepop.worker.pipeline_backend import MAX_OPEN_SOURCES
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_syncify import SyncWorkerEndpoint
from eyepop.worker.worker_types import Pop

log = logging.getLogger('eyepop')
log.debug(f"EyePop SDK v{__version__} initializing...")

def _pipeline_endpoint(
        pop_id: str | None,
        session_uuid: str | None,
        secret_key: str | None,
        api_key: str | None,
        access_token: str | None,
        eyepop_url: str | None,
        job_queue_length: int | None,
        is_local_mode: bool,
        dataset_uuid: str | None,
        pipeline_image: str | None,
        pipeline_version: str | None,
        session_name: str | None,
        pop: Pop | dict[str, object] | None,
        account_id: str | None,
) -> WorkerEndpoint:
    """A worker endpoint whose jobs run on the pipeline API (SDK-24).

    The credential comes from the environment as for a worker session, except
    that a secret key does not count: pipeline mode takes an API key or a user
    access token. The endpoint refuses what only a worker session has.
    """
    if pop_id is None:
        pop_id = os.getenv("EYEPOP_POP_ID", "transient")
    if session_uuid is None:
        session_uuid = os.getenv("EYEPOP_SESSION_UUID")

    if access_token is None and api_key is None and secret_key is None and not is_local_mode:
        api_key = os.getenv("EYEPOP_API_KEY")
        if api_key is None:
            access_token = os.getenv("EYEPOP_ACCESS_TOKEN")
        if api_key is None and access_token is None:
            raise KeyError("pipeline mode needs EYEPOP_API_KEY, EYEPOP_ACCESS_TOKEN, api_key or access_token")

    if job_queue_length is None:
        job_queue_length = MAX_OPEN_SOURCES

    if eyepop_url is None:
        eyepop_url = os.getenv("EYEPOP_URL")
        if eyepop_url is None:
            # Local mode talks to the pipeline API itself, on its default listener.
            eyepop_url = 'http://127.0.0.1:9002' if is_local_mode else settings.default_compute_url

    return WorkerEndpoint(
        secret_key=secret_key,
        access_token=access_token,
        api_key=api_key,
        eyepop_url=eyepop_url,
        pop_id=pop_id,
        session_uuid=session_uuid,
        auto_start=True,
        stop_jobs=False,
        job_queue_length=job_queue_length,
        request_tracer_max_buffer=0,
        dataset_uuid=dataset_uuid,
        pipeline_image=pipeline_image,
        pipeline_version=pipeline_version,
        session_name=session_name,
        pop=pop,
        is_local_mode=is_local_mode,
        account_id=account_id,
        is_pipeline_mode=True,
    )


class EyePopSdk:
    """EyePop.ai Python SDK for Worker API."""

    @deprecated("use EyePopSdk.sync_worker() or EyePopSdk.async_worker() instead")
    @staticmethod
    def workerEndpoint(
            pop_id: str | None = None,
            session_uuid: str | None = None,
            secret_key: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            auto_start: bool = True,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int | None = None,
            is_async: bool = False,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
            session_name: str | None = None,
            pop: Pop | dict[str, object] | None = None,
            account_id: str | None = None,
            is_pipeline_mode: bool | None = None,
    ) -> WorkerEndpoint | SyncWorkerEndpoint:
        if is_async:
            return EyePopSdk.async_worker(
                pop_id=pop_id,
                session_uuid=session_uuid,
                secret_key=secret_key,
                api_key=api_key,
                access_token=access_token,
                auto_start=auto_start,
                stop_jobs=stop_jobs,
                eyepop_url=eyepop_url,
                job_queue_length=job_queue_length,
                is_local_mode=is_local_mode,
                request_tracer_max_buffer=request_tracer_max_buffer,
                dataset_uuid=dataset_uuid,
                pipeline_image=pipeline_image,
                pipeline_version=pipeline_version,
                session_name=session_name,
                pop=pop,
                account_id=account_id,
                is_pipeline_mode=is_pipeline_mode,
            )
        else:
            return EyePopSdk.sync_worker(
                pop_id=pop_id,
                session_uuid=session_uuid,
                secret_key=secret_key,
                api_key=api_key,
                access_token=access_token,
                auto_start=auto_start,
                stop_jobs=stop_jobs,
                eyepop_url=eyepop_url,
                job_queue_length=job_queue_length,
                is_local_mode=is_local_mode,
                request_tracer_max_buffer=request_tracer_max_buffer,
                dataset_uuid=dataset_uuid,
                pipeline_image=pipeline_image,
                pipeline_version=pipeline_version,
                session_name=session_name,
                pop=pop,
                account_id=account_id,
                is_pipeline_mode=is_pipeline_mode,
            )

    @staticmethod
    def sync_worker(
            pop_id: str | None = None,
            session_uuid: str | None = None,
            secret_key: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            auto_start: bool = True,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int | None = None,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
            session_name: str | None = None,
            pop: Pop | dict[str, object] | None = None,
            account_id: str | None = None,
            is_pipeline_mode: bool | None = None,
    ) -> SyncWorkerEndpoint:
        endpoint = EyePopSdk.async_worker(
            pop_id=pop_id,
            session_uuid=session_uuid,
            secret_key=secret_key,
            api_key=api_key,
            access_token=access_token,
            auto_start=auto_start,
            stop_jobs=stop_jobs,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            is_local_mode=is_local_mode,
            request_tracer_max_buffer=request_tracer_max_buffer,
            dataset_uuid=dataset_uuid,
            pipeline_image=pipeline_image,
            pipeline_version=pipeline_version,
            session_name=session_name,
            pop=pop,
            account_id=account_id,
            is_pipeline_mode=is_pipeline_mode,
        )
        return SyncWorkerEndpoint(endpoint)

    @staticmethod
    def async_worker(
            pop_id: str | None = None,
            session_uuid: str | None = None,
            secret_key: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            auto_start: bool = True,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int | None = None,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
            session_name: str | None = None,
            pop: Pop | dict[str, object] | None = None,
            account_id: str | None = None,
            is_pipeline_mode: bool | None = None,
    ) -> WorkerEndpoint:
        """A worker endpoint.

        With `is_pipeline_mode`, or `EYEPOP_PIPELINE_MODE` set to true, its
        jobs run on the pipeline API's socket rather than on a worker session;
        the compute API at `eyepop_url` names the pipeline API in its
        `/v1/configs`. Pipeline mode takes an API key or a user access token,
        and no option that names a session or a pop.
        """
        if is_local_mode is None:
            local_mode_env = os.getenv("EYEPOP_LOCAL_MODE", "")
            is_local_mode = local_mode_env.lower() in ("true", "yes")

        if is_pipeline_mode is None:
            is_pipeline_mode = os.getenv("EYEPOP_PIPELINE_MODE", "").lower() in ("true", "yes", "1")

        if is_pipeline_mode:
            return _pipeline_endpoint(
                pop_id=pop_id,
                session_uuid=session_uuid,
                secret_key=secret_key,
                api_key=api_key,
                access_token=access_token,
                eyepop_url=eyepop_url,
                job_queue_length=job_queue_length,
                is_local_mode=is_local_mode,
                dataset_uuid=dataset_uuid,
                pipeline_image=pipeline_image,
                pipeline_version=pipeline_version,
                session_name=session_name,
                pop=pop,
                account_id=account_id,
            )

        if job_queue_length is None:
            job_queue_length = settings.default_job_queue_length

        if pop_id is None:
            pop_id = os.getenv("EYEPOP_POP_ID", "transient")

        if session_uuid is None:
            session_uuid = os.getenv("EYEPOP_SESSION_UUID", None)

        has_any_auth_key = access_token is not None or secret_key is not None or api_key is not None

        if not has_any_auth_key and not is_local_mode:
            secret_key = os.getenv("EYEPOP_SECRET_KEY")
            api_key = os.getenv("EYEPOP_API_KEY")
            if secret_key is None and api_key is None:
                raise KeyError(
                    "At least one authentication method required: "
                    "EYEPOP_SECRET_KEY or EYEPOP_API_KEY or access_token"
                )

        if eyepop_url is None:
            eyepop_url = os.getenv("EYEPOP_URL")
            if eyepop_url is None:
                if is_local_mode:
                    eyepop_url = 'http://127.0.0.1:8080'
                elif api_key:
                    eyepop_url = "https://compute.eyepop.ai"
                else:
                    eyepop_url = "https://api.eyepop.ai"

        is_transient_pop = pop_id == "transient"

        if api_key and not is_transient_pop:
            raise ValueError(
                f"EYEPOP_API_KEY can only be used with transient pops. "
                f"Current pop_id: '{pop_id}'. Use EYEPOP_SECRET_KEY for named pops."
            )

        is_compute_url = eyepop_url and "https://compute" in eyepop_url.lower()
        if is_compute_url:
            if not api_key:
                raise ValueError(f"Compute API endpoint ({eyepop_url}) requires EYEPOP_API_KEY")
            if not is_transient_pop:
                raise ValueError(f"Compute API only supports transient mode. Current pop_id: '{pop_id}'")

        if is_local_mode and api_key is None:
            api_key = "<local api key>"

        assert eyepop_url
        log.debug(f"EyePop URL: {eyepop_url}")

        endpoint = WorkerEndpoint(
            secret_key=secret_key,
            access_token=access_token,
            api_key=api_key,
            pop_id=pop_id,
            session_uuid=session_uuid,
            auto_start=auto_start,
            stop_jobs=stop_jobs,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            dataset_uuid=dataset_uuid,
            pipeline_image=pipeline_image,
            pipeline_version=pipeline_version,
            session_name=session_name,
            pop=pop,
            account_id=account_id,
            is_local_mode=is_local_mode,
        )
        return endpoint

    """
    EyePop.ai Python SDK for Data API
    """

    @staticmethod
    def dataEndpoint(
        account_id: str | None = None,
        secret_key: str | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
        eyepop_url: str | None = None,
        job_queue_length: int = 1024,
        is_async: bool = False,
        request_tracer_max_buffer: int = 1204,
        disable_ws: bool = True,
    ) -> DataEndpoint | SyncDataEndpoint:
        if access_token is None and secret_key is None and api_key is None:
            secret_key = os.getenv("EYEPOP_SECRET_KEY")
            api_key = os.getenv("EYEPOP_API_KEY")
            if secret_key is None and api_key is None:
                raise KeyError(
                    "At least one authentication method required: "
                    "EYEPOP_SECRET_KEY or EYEPOP_API_KEY or access_token"
                )

        if eyepop_url is None:
            eyepop_url = os.getenv("EYEPOP_URL")
            if eyepop_url is None:
                if api_key:
                    eyepop_url = "https://compute.eyepop.ai"
                else:
                    eyepop_url = "https://api.eyepop.ai"

        if account_id is None:
            account_id = account_uuid_from_env()

        endpoint = DataEndpoint(
            secret_key=secret_key,
            access_token=access_token,
            account_id=account_id,
            api_key=api_key,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            disable_ws=disable_ws,
        )

        if not is_async:
            return SyncDataEndpoint(endpoint)

        return endpoint

    from matplotlib.axes import Axes
    @staticmethod
    def plot(axes: Axes):
        from eyepop.visualize import EyePopPlot
        return EyePopPlot(axes)
