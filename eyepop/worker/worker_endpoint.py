import logging
from typing import Any, AsyncIterable, BinaryIO, Callable

import aiohttp

from eyepop.data.types.asset import Area
from eyepop.endpoint import Endpoint
from eyepop.settings import account_uuid_from_env
from eyepop.worker.camera import Camera
from eyepop.worker.pipeline_backend import MAX_OPEN_SOURCES, PipelineBackend
from eyepop.worker.session_backend import SessionBackend
from eyepop.worker.worker_backend import WorkerBackend
from eyepop.worker.worker_client_session import WorkerClientSession
from eyepop.worker.worker_jobs import WorkerJob, WorkerSource, _file_upload, _stream_upload
from eyepop.worker.worker_types import (
    DEFAULT_PREDICTION_VERSION,
    ComponentParams,
    MotionDetectConfig,
    Pop,
    PredictionVersion,
    VideoMode,
)

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
    """Endpoint to an EyePop.ai worker.

    Its jobs run on a backend: a worker session by default, or with
    `is_pipeline_mode` the pipeline API's socket. The methods are the same
    either way; a backend that cannot honour an option says so.
    """

    def __init__(
            self,
            secret_key: str | None,
            access_token: str | None,
            api_key: str | None,
            eyepop_url: str,
            pop_id: str,
            session_uuid: str | None,
            auto_start: bool,
            stop_jobs: bool,
            job_queue_length: int,
            request_tracer_max_buffer: int,
            dataset_uuid: str | None = None,
            pipeline_image: str | None = None,
            pipeline_version: str | None = None,
            session_name: str | None = None,
            pop: Pop | dict[str, Any] | None = None,
            is_local_mode: bool = False,
            account_id: str | None = None,
            *,
            is_pipeline_mode: bool = False,
    ):
        if is_pipeline_mode:
            _refuse_session_options(
                secret_key=secret_key,
                pop_id=pop_id if pop_id != 'transient' else None,
                session_uuid=session_uuid,
                pipeline_image=pipeline_image,
                pipeline_version=pipeline_version,
                session_name=session_name,
                dataset_uuid=dataset_uuid,
            )
            if job_queue_length > MAX_OPEN_SOURCES:
                raise ValueError(
                    f'job_queue_length {job_queue_length} exceeds {MAX_OPEN_SOURCES}, '
                    f'the open sources a pipeline API socket takes')
            # The request tracer posts to a worker's /events, which neither the
            # compute API nor the pipeline API serves.
            request_tracer_max_buffer = 0
        super().__init__(
            secret_key=secret_key,
            access_token=access_token,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            request_tracer_max_buffer=request_tracer_max_buffer,
            api_key=api_key,
            session_uuid=session_uuid,
        )
        self.is_local_mode = is_local_mode
        self.pop_id = pop_id
        self.auto_start = auto_start
        self.stop_jobs = stop_jobs
        self.dataset_uuid = dataset_uuid
        self.pop = pop if isinstance(pop, Pop) else Pop(**pop) if pop is not None else None
        if self.compute_ctx:
            # The environment is read only when no account is named, so a deprecated
            # EYEPOP_ACCOUNT_ID cannot warn (or, with warnings as errors, fail) for a
            # caller who named one.
            self.compute_ctx.account_uuid = account_id if account_id is not None else account_uuid_from_env()

        self.is_pipeline_mode = is_pipeline_mode
        self._backend: WorkerBackend
        if is_pipeline_mode:
            self._backend = PipelineBackend(self)
        else:
            self._backend = SessionBackend(self, pipeline_image, pipeline_version, session_name)

    @property
    def backend(self) -> WorkerBackend:
        """What runs this endpoint's jobs."""
        return self._backend

    async def _reconnect(self):
        await self._backend.connect()

    async def _disconnect(self, timeout: float | None = None):
        await self._backend.disconnect(timeout)

    async def access_session(self) -> dict | None:
        """The endpoint's credential as a session, before its backend adds what it runs on."""
        return await Endpoint.session(self)

    # Parent returns dict | None — must match and guard against None before subscripting
    async def session(self) -> dict | None:
        return await self._backend.session()

    async def get_pop(self) -> Pop | None:
        return self.pop

    async def set_pop(self, pop: Pop):
        return await self._backend.set_pop(pop)

    async def dev_mode_base_url(self) -> str:
        return await self._backend.dev_mode_base_url()

    async def dev_mode_pipeline_base_url(self) -> str:
        return await self._backend.dev_mode_pipeline_base_url()

    async def upload(
            self,
            location: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None,
            # Keyword-only: this method published its positional order without
            # them, and `params` must keep its slot.
            *,
            is_live: bool | None = None,
    ) -> WorkerJob:
        return await self._submit(WorkerSource(
            uploads=[_file_upload(location)],
            video_mode=video_mode,
            is_live=is_live,
            component_params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            version=self._prediction_version(),
        ), on_ready)

    async def upload_stream(
            self,
            stream: BinaryIO | AsyncIterable[bytes],
            mime_type: str,
            video_mode: VideoMode | None = None,
            is_live: bool | None = None,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        return await self._submit(WorkerSource(
            uploads=[_stream_upload(stream, mime_type)],
            video_mode=video_mode,
            is_live=is_live,
            component_params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            version=self._prediction_version(),
        ), on_ready)

    async def upload_stream_group(
            self,
            streams: list[BinaryIO],
            mime_types: list[str] | None = None,
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        """Uploads multiple in-memory streams as a single image group (one inference unit).

        The streams are processed together and yield a single prediction stream;
        order follows `streams`. `mime_types`, if given, is a parallel list setting
        each stream's content type and must match `streams` in length; if omitted,
        no per-stream content type is sent. `params`, `roi`, and
        `media_cache_seconds` apply once to the whole group.
        """
        if not streams:
            raise ValueError("upload requires at least one source")
        if mime_types is not None and len(mime_types) != len(streams):
            raise ValueError("mime_types must have the same length as streams")
        types: list[str | None] = list(mime_types) if mime_types is not None else [None] * len(streams)
        return await self._submit(WorkerSource(
            uploads=[_stream_upload(stream, mime_type) for stream, mime_type in zip(streams, types, strict=True)],
            group=True,
            component_params=params,
            roi=roi,
            camera=camera,
            media_cache_seconds=media_cache_seconds,
            version=self._prediction_version(),
        ), on_ready)

    async def upload_group(
            self,
            locations: list[str],
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        """Uploads multiple local images as a single image group (one inference unit).

        The images are processed together and yield a single prediction stream.
        Image order follows `locations`. Unlike `upload`, there is no `video_mode`
        or `motion_detect` (a group is still images); `params`, `roi`, and
        `media_cache_seconds` apply once to the whole group.
        """
        if not locations:
            raise ValueError("upload requires at least one source")
        return await self._submit(WorkerSource(
            uploads=[_file_upload(location) for location in locations],
            group=True,
            component_params=params,
            roi=roi,
            camera=camera,
            media_cache_seconds=media_cache_seconds,
            version=self._prediction_version(),
        ), on_ready)

    async def load_from(
            self,
            location: str,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            rtsp_force_non_compliant_url: bool | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        """Loads a server-fetched URL as a single source.

        `rtsp_force_non_compliant_url` applies to an `rtsp:` location only, and
        only for servers that require the pre-RFC-2326 SETUP URL. Leave it unset
        unless a camera fails to open without it: the compliant construction is
        what a camera advertising an absolute control URL needs, and forcing the
        old one makes such a camera answer 404.
        """
        return await self._submit(WorkerSource(
            urls=[location],
            component_params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            rtsp_force_non_compliant_url=rtsp_force_non_compliant_url,
            version=self._prediction_version(),
        ), on_ready)

    async def load_from_group(
            self,
            locations: list[str],
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        """Loads multiple server-fetched URLs as a single image group (one inference unit).

        The URLs are processed together and yield a single prediction stream;
        order follows `locations`. The server fetches each URL and reads its own
        content type, so no mime type is supplied. `params`, `roi`, and
        `media_cache_seconds` apply once to the whole group.
        """
        if not locations:
            raise ValueError("load_from requires at least one url")
        return await self._submit(WorkerSource(
            urls=list(locations),
            group=True,
            component_params=params,
            roi=roi,
            camera=camera,
            media_cache_seconds=media_cache_seconds,
            version=self._prediction_version(),
        ), on_ready)

    async def load_asset(
            self,
            asset_uuid: str,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        return await self._submit(WorkerSource(
            asset_uuid=asset_uuid,
            component_params=params,
            motion_detect=motion_detect,
            roi=roi,
            camera=camera,
            fps=fps,
            media_cache_seconds=media_cache_seconds,
            version=self._prediction_version(),
        ), on_ready)

    async def _submit(self, source: WorkerSource, on_ready: Callable[[WorkerJob], None] | None) -> WorkerJob:
        self._backend.accept(source)
        job = WorkerJob(self, self._backend, source, on_ready=on_ready, callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    def _prediction_version(self) -> PredictionVersion:
        # A select forward's results come only at V3. Only a Pop that selects
        # asks for it, so a worker too old to know V3 - which also rejects
        # that Pop - is never sent a version it would not accept.
        if self.pop is not None and self.pop.selects():
            return PredictionVersion.V3
        return DEFAULT_PREDICTION_VERSION

    #
    # Implements: WorkerClientSession, on the backend's worker session
    #

    async def pipeline_get(self, url_path_and_query: str, accept: str | None = None,
                           timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._backend.pipeline_get(url_path_and_query, accept=accept, timeout=timeout)

    async def pipeline_post(self, url_path_and_query: str, accept: str | None = None, open_data: Callable | None = None,
                            content_type: str | None = None,
                            timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._backend.pipeline_post(url_path_and_query, accept=accept, open_data=open_data,
                                                 content_type=content_type, timeout=timeout)

    async def pipeline_patch(self, url_path_and_query: str, accept: str | None = None, data: Any = None,
                             content_type: str | None = None,
                             timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._backend.pipeline_patch(url_path_and_query, accept=accept, data=data,
                                                  content_type=content_type, timeout=timeout)

    async def pipeline_delete(self, url_path_and_query: str,
                              timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        return await self._backend.pipeline_delete(url_path_and_query, timeout=timeout)


def _refuse_session_options(**options: Any):
    """Pipeline mode runs no worker session: an option that names one, or what only a session does, is refused."""
    named = [name for name, value in options.items() if value]
    if not named:
        return
    if named == ['dataset_uuid']:
        raise ValueError("pipeline mode does not write predictions to a dataset yet (AWSU-316); "
                         "it takes no dataset_uuid")
    if named == ['secret_key']:
        raise ValueError("pipeline mode takes an API key or a user access token, not a secret key")
    raise ValueError(f"pipeline mode runs no worker session; it takes no {', '.join(named)}")
