import logging
import os

from eyepop import __version__
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_syncify import SyncDataEndpoint
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_syncify import SyncWorkerEndpoint

log = logging.getLogger('eyepop')
log.debug(f"EyePop SDK v{__version__} initializing...")

class EyePopSdk:
    """EyePop.ai Python SDK for Worker API."""

    @staticmethod
    def workerEndpoint(
            pop_id: str | None = None,
            secret_key: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            auto_start: bool = True,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int = 1024,
            is_async: bool = False,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None
    ) -> WorkerEndpoint | SyncWorkerEndpoint:
        if is_local_mode is None:
            local_mode_env = os.getenv("EYEPOP_LOCAL_MODE", "")
            is_local_mode = local_mode_env.lower() in ("true", "yes")

        if eyepop_url is None:
            if is_local_mode:
                eyepop_url = 'http://127.0.0.1:8080'
            else:
                eyepop_url = os.getenv("EYEPOP_URL", "https://api.eyepop.ai")

        if pop_id is None:
            pop_id = os.getenv("EYEPOP_POP_ID", "transient")

        has_any_auth_key = access_token is not None or secret_key is not None or api_key is not None

        if not has_any_auth_key and not is_local_mode:
            secret_key = os.getenv("EYEPOP_SECRET_KEY")
            api_key = os.getenv("EYEPOP_API_KEY")
            if secret_key is None and api_key is None:
                raise KeyError(
                    "At least one authentication method required: "
                    "EYEPOP_SECRET_KEY or EYEPOP_API_KEY or access_token"
                )

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

        log.debug(f"Eyepop URL: {eyepop_url}")

        endpoint = WorkerEndpoint(
            secret_key=secret_key,
            access_token=access_token,
            api_key=api_key,
            pop_id=pop_id,
            auto_start=auto_start,
            stop_jobs=stop_jobs,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            dataset_uuid=dataset_uuid,
        )

        if not is_async:
            return SyncWorkerEndpoint(endpoint)

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
                eyepop_url = "https://api.eyepop.ai"

        if account_id is None:
            account_id = os.getenv("EYEPOP_ACCOUNT_ID")

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
