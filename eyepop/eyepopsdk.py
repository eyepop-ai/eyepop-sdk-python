import logging
import os

from typing_extensions import deprecated

from eyepop import __version__
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_syncify import SyncDataEndpoint
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_syncify import SyncWorkerEndpoint

log = logging.getLogger('eyepop')
log.debug(f"EyePop SDK v{__version__} initializing...")


_SECRET_KEY_REMOVED_MSG = (
    "EYEPOP_SECRET_KEY / secret_key authentication has been removed. "
    "Use EYEPOP_API_KEY (or pass api_key=) and create deployments via the "
    "EyePop dashboard or CLI when you need a named, persistent session."
)

_POP_ID_DEPRECATED_MSG = (
    "pop_id is deprecated. The SDK no longer provisions named pops; "
    "create a deployment via the EyePop dashboard or CLI and pass its "
    "session_uuid to the SDK instead. See the deployment docs."
)


def _reject_secret_key(secret_key: str | None) -> None:
    if secret_key is not None or os.getenv("EYEPOP_SECRET_KEY"):
        raise ValueError(_SECRET_KEY_REMOVED_MSG)


def _warn_if_named_pop(pop_id: str | None) -> None:
    if pop_id and pop_id != "transient":
        log.warning(_POP_ID_DEPRECATED_MSG)


class EyePopSdk:
    """EyePop.ai Python SDK for Worker API."""

    @deprecated("use EyePopSdk.sync_worker() or EyePopSdk.async_worker() instead")
    @staticmethod
    def workerEndpoint(
            pop_id: str | None = None,
            session_uuid: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int = 1024,
            is_async: bool = False,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
    ) -> WorkerEndpoint | SyncWorkerEndpoint:
        if is_async:
            return EyePopSdk.async_worker(
                pop_id=pop_id,
                session_uuid=session_uuid,
                api_key=api_key,
                access_token=access_token,
                stop_jobs=stop_jobs,
                eyepop_url=eyepop_url,
                job_queue_length=job_queue_length,
                is_local_mode=is_local_mode,
                request_tracer_max_buffer=request_tracer_max_buffer,
                dataset_uuid=dataset_uuid,
                pipeline_image=pipeline_image,
                pipeline_version=pipeline_version,
            )
        else:
            return EyePopSdk.sync_worker(
                pop_id=pop_id,
                session_uuid=session_uuid,
                api_key=api_key,
                access_token=access_token,
                stop_jobs=stop_jobs,
                eyepop_url=eyepop_url,
                job_queue_length=job_queue_length,
                is_local_mode=is_local_mode,
                request_tracer_max_buffer=request_tracer_max_buffer,
                dataset_uuid=dataset_uuid,
                pipeline_image=pipeline_image,
                pipeline_version=pipeline_version,
            )

    @staticmethod
    def sync_worker(
            pop_id: str | None = None,
            session_uuid: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int = 1024,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
            secret_key: str | None = None,
    ) -> SyncWorkerEndpoint:
        _reject_secret_key(secret_key)
        endpoint = EyePopSdk.async_worker(
            pop_id=pop_id,
            session_uuid=session_uuid,
            api_key=api_key,
            access_token=access_token,
            stop_jobs=stop_jobs,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            is_local_mode=is_local_mode,
            request_tracer_max_buffer=request_tracer_max_buffer,
            dataset_uuid=dataset_uuid,
            pipeline_image=pipeline_image,
            pipeline_version=pipeline_version,
        )
        return SyncWorkerEndpoint(endpoint)

    @staticmethod
    def async_worker(
            pop_id: str | None = None,
            session_uuid: str | None = None,
            api_key: str | None = None,
            access_token: str | None = None,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int = 1024,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
            secret_key: str | None = None,
    ) -> WorkerEndpoint:
        _reject_secret_key(secret_key)

        if is_local_mode is None:
            local_mode_env = os.getenv("EYEPOP_LOCAL_MODE", "")
            is_local_mode = local_mode_env.lower() in ("true", "yes")

        if pop_id is None:
            pop_id = os.getenv("EYEPOP_POP_ID", "transient")

        if session_uuid is None:
            session_uuid = os.getenv("EYEPOP_SESSION_UUID", None)

        _warn_if_named_pop(pop_id)

        if access_token is None and api_key is None and not is_local_mode:
            api_key = os.getenv("EYEPOP_API_KEY")
            if api_key is None:
                raise KeyError(
                    "Authentication required: set EYEPOP_API_KEY or pass "
                    "api_key= or access_token="
                )

        if eyepop_url is None:
            eyepop_url = os.getenv("EYEPOP_URL")
            if eyepop_url is None:
                eyepop_url = "http://127.0.0.1:8080" if is_local_mode else "https://compute.eyepop.ai"

        if is_local_mode and api_key is None:
            api_key = "<local api key>"

        assert eyepop_url
        log.debug(f"EyePop URL: {eyepop_url}")

        endpoint = WorkerEndpoint(
            access_token=access_token,
            api_key=api_key,
            pop_id=pop_id,
            session_uuid=session_uuid,
            stop_jobs=stop_jobs,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            dataset_uuid=dataset_uuid,
            pipeline_image=pipeline_image,
            pipeline_version=pipeline_version,
        )
        return endpoint

    """
    EyePop.ai Python SDK for Data API
    """

    @staticmethod
    def dataEndpoint(
        account_id: str | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
        eyepop_url: str | None = None,
        job_queue_length: int = 1024,
        is_async: bool = False,
        request_tracer_max_buffer: int = 1204,
        disable_ws: bool = True,
        secret_key: str | None = None,
    ) -> DataEndpoint | SyncDataEndpoint:
        _reject_secret_key(secret_key)

        if access_token is None and api_key is None:
            api_key = os.getenv("EYEPOP_API_KEY")
            if api_key is None:
                raise KeyError(
                    "Authentication required: set EYEPOP_API_KEY or pass "
                    "api_key= or access_token="
                )

        if eyepop_url is None:
            eyepop_url = os.getenv("EYEPOP_URL", "https://compute.eyepop.ai")

        if account_id is None:
            account_id = os.getenv("EYEPOP_ACCOUNT_ID")

        endpoint = DataEndpoint(
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
