import logging
import os
import sys
import warnings

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

# EYEPOP_ENV names the pipeline API, and the compute API that mints the token it takes.
_PIPELINE_API_URLS = {
    "staging": "https://pipeline.staging.eyepop.xyz",
    "production": "https://pipeline.eyepop.ai",
    "local": "http://127.0.0.1:9002",
}
_COMPUTE_API_URLS = {
    "staging": "https://compute.staging.eyepop.xyz",
    "production": "https://compute.eyepop.ai",
}

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
        pipeline_api_url: str | None,
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

    env_name = (os.getenv("EYEPOP_ENV") or "").strip().lower()
    # A compute API named by the caller names the environment: EYEPOP_ENV is read only when none is, so a
    # token minted by one environment never goes to the pipeline API of another.
    compute_named = eyepop_url is not None or os.getenv("EYEPOP_URL") is not None
    if env_name == "local" and not compute_named:
        # A local environment is local mode: no compute API mints a token for it.
        is_local_mode = True
    if env_name and not is_local_mode and not compute_named:
        if env_name not in _PIPELINE_API_URLS:
            raise ValueError(f"EYEPOP_ENV={env_name} is not one of {', '.join(_PIPELINE_API_URLS)}")
        eyepop_url = _COMPUTE_API_URLS[env_name]
        if pipeline_api_url is None:
            pipeline_api_url = _PIPELINE_API_URLS[env_name]

    if access_token is None and api_key is None and secret_key is None and not is_local_mode:
        api_key = os.getenv("EYEPOP_API_KEY")
        if api_key is None:
            access_token = os.getenv("EYEPOP_ACCESS_TOKEN")
        if api_key is None and access_token is None:
            raise KeyError("pipeline mode needs EYEPOP_API_KEY, EYEPOP_ACCESS_TOKEN, api_key or access_token (EYEPOP_SECRET_KEY is for a worker session: pass is_pipeline_mode=False or set EYEPOP_PIPELINE_MODE=false)")

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
        pipeline_api_url=pipeline_api_url,
    )


# The file whose wrapper `deprecated` puts between workerEndpoint and its caller: typing_extensions, or the
# standard library's warnings module from Python 3.13.
_NOT_THE_CALLER = (os.path.dirname(__file__) + os.sep, deprecated.__call__.__code__.co_filename)


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
            pipeline_api_url: str | None = None,
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
                pipeline_api_url=pipeline_api_url,
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
                pipeline_api_url=pipeline_api_url,
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
            pipeline_api_url: str | None = None,
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
            pipeline_api_url=pipeline_api_url,
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
            pipeline_api_url: str | None = None,
    ) -> WorkerEndpoint:
        """A worker endpoint.

        Its jobs run on the pipeline API's socket rather than on a worker
        session, unless `is_pipeline_mode` is False or `EYEPOP_PIPELINE_MODE`
        is false;
        `pipeline_api_url` or `EYEPOP_ENV` names the pipeline API; unset, the
        compute API at `eyepop_url` names it in its `/v1/configs`. Pipeline mode takes an API key or a user access token,
        and no option that names a session or a pop. A worker session is
        deprecated and warns with a `DeprecationWarning`.
        """
        if is_local_mode is None:
            local_mode_env = os.getenv("EYEPOP_LOCAL_MODE", "")
            is_local_mode = local_mode_env.lower() in ("true", "yes")

        if is_pipeline_mode is None:
            is_pipeline_mode = os.getenv("EYEPOP_PIPELINE_MODE", "").strip().lower() not in ("false", "no", "0")

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
                pipeline_api_url=pipeline_api_url,
            )

        if pipeline_api_url is not None:
            raise ValueError("pipeline_api_url names the pipeline API, which only pipeline mode uses; "
                             "drop it, or leave is_pipeline_mode and EYEPOP_PIPELINE_MODE at their defaults")

        # The warning has to blame the caller's line, not an SDK frame or the wrapper of
        # the deprecated workerEndpoint: Python shows a DeprecationWarning only when the blamed module is __main__.
        level, frame = 1, sys._getframe(0)
        while frame is not None and frame.f_code.co_filename.startswith(_NOT_THE_CALLER):
            level, frame = level + 1, frame.f_back
        warnings.warn("worker sessions (is_pipeline_mode=False) are deprecated, use pipeline mode, the default",
                      DeprecationWarning, stacklevel=level)

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
                else:
                    eyepop_url = "https://compute.eyepop.ai"

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
    def async_data(
        account_id: str | None = None,
        secret_key: str | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
        eyepop_url: str | None = None,
        job_queue_length: int = 1024,
        request_tracer_max_buffer: int = 1204,
        disable_ws: bool = True,
    ) -> DataEndpoint:
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
                eyepop_url = "https://compute.eyepop.ai"

        if account_id is None:
            account_id = account_uuid_from_env()

        return DataEndpoint(
            secret_key=secret_key,
            access_token=access_token,
            account_id=account_id,
            api_key=api_key,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            disable_ws=disable_ws,
        )

    @staticmethod
    def sync_data(
        account_id: str | None = None,
        secret_key: str | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
        eyepop_url: str | None = None,
        job_queue_length: int = 1024,
        request_tracer_max_buffer: int = 1204,
        disable_ws: bool = True,
    ) -> SyncDataEndpoint:
        return SyncDataEndpoint(EyePopSdk.async_data(
            account_id=account_id,
            secret_key=secret_key,
            access_token=access_token,
            api_key=api_key,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            disable_ws=disable_ws,
        ))

    @deprecated("use EyePopSdk.sync_data() or EyePopSdk.async_data() instead")
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
        endpoint = EyePopSdk.async_data(
            account_id=account_id,
            secret_key=secret_key,
            access_token=access_token,
            api_key=api_key,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            disable_ws=disable_ws,
        )
        return endpoint if is_async else SyncDataEndpoint(endpoint)

    from matplotlib.axes import Axes
    @staticmethod
    def plot(axes: Axes):
        from eyepop.visualize import EyePopPlot
        return EyePopPlot(axes)
