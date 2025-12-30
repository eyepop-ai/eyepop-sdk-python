import json
import time
from asyncio import Queue
from typing import Any, BinaryIO, Callable, Sequence
from urllib.parse import quote_plus

import aiohttp
from pydantic import BaseModel, Field

from eyepop.client_session import ClientSession
from eyepop.data.data_types import Asset, AssetImport, InferRequest, Prediction, EvaluateRequest, EvaluateResponse, \
    InferRunInfo, APPLICATION_JSON
from eyepop.jobs import Job, JobStateCallback


class DataJob(Job):
    """Abstract Job submitted to an EyePop.ai DataEndpoint."""
    timeout: aiohttp.ClientTimeout | None

    def __init__(
            self, session: ClientSession,
            on_ready: Callable[["DataJob"], None] | None,
            callback: JobStateCallback | None = None,
            timeout: aiohttp.ClientTimeout | None = aiohttp.ClientTimeout(total=None, sock_read=60)
    ):
        super().__init__(session, on_ready, callback)
        self.timeout = timeout

    async def result(self) -> Asset:
        return await self.pop_result()

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        raise NotImplementedError("can't execute abstract jobs")


class _UploadStreamJob(DataJob):
    def __init__(
            self,
            stream: BinaryIO | Callable[[], Any],
            mime_type: str,
            dataset_uuid: str,
            dataset_version: int | None,
            external_id: str | None,
            sync_transform: bool | None,
            no_transform: bool | None,
            session: ClientSession,
            on_ready: Callable[[DataJob], None] | None = None,
            callback: JobStateCallback | None = None,
            timeout: aiohttp.ClientTimeout | None = aiohttp.ClientTimeout(total=None, sock_read=60)
    ):
        super().__init__(session, on_ready, callback, timeout)
        self.stream = stream
        self.mime_type = mime_type
        self.dataset_uuid = dataset_uuid
        self.dataset_version = dataset_version
        self.external_id = external_id
        self.sync_transform = sync_transform
        self.no_transform = no_transform

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        dataset_version_query = f"&dataset_version={self.dataset_version}" if self.dataset_version else ""
        external_id_query = f"&external_id={quote_plus(self.external_id)}" if self.external_id else ""

        post_path = f"/assets?dataset_uuid={self.dataset_uuid}{dataset_version_query}{external_id_query}"

        if self.sync_transform is not None:
            post_path = f"{post_path}&sync_transform={'true' if self.sync_transform else 'false'}"
        if self.no_transform is not None:
            post_path = f"{post_path}&no_transform={'true' if self.no_transform else 'false'}"

        async with await session.request_with_retry(
                method="POST",
                url=post_path,
                data=self.stream,
                content_type=self.mime_type,
                timeout=self.timeout
        ) as resp:
            result = Asset.model_validate(await resp.json())
            await queue.put(result)


class _ImportFromJob(DataJob):
    def __init__(
            self,
            asset_import: AssetImport,
            dataset_uuid: str,
            dataset_version: int | None,
            external_id: str | None,
            partition: str | None,
            sync_transform: bool | None,
            no_transform: bool | None,
            session: ClientSession,
            on_ready: Callable[[DataJob], None] | None = None,
            callback: JobStateCallback | None = None,
            timeout: aiohttp.ClientTimeout | None = aiohttp.ClientTimeout(total=None, sock_read=60)
    ):
        super().__init__(session, on_ready, callback, timeout)
        self.asset_import = asset_import
        self.dataset_uuid = dataset_uuid
        self.dataset_version = dataset_version
        self.external_id = external_id
        self.partition = partition
        self.sync_transform = sync_transform
        self.no_transform = no_transform

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        dataset_version_query = f"&dataset_version={self.dataset_version}" if self.dataset_version else ""
        external_id_query = f"&external_id={quote_plus(self.external_id)}" if self.external_id else ""
        partition_query = f"&partition={quote_plus(self.partition)}" if self.partition else ""

        post_path = (f"/assets/imports?dataset_uuid={self.dataset_uuid}"
                     f"{dataset_version_query}{external_id_query}{partition_query}")

        if self.sync_transform is not None:
            post_path = f"{post_path}&sync_transform={'true' if self.sync_transform else 'false'}"
        if self.no_transform is not None:
            post_path = f"{post_path}&no_transform={'true' if self.no_transform else 'false'}"

        async with await session.request_with_retry(
                "POST",
                post_path,
                data=self.asset_import.model_dump_json(exclude_unset=True),
                content_type="application/json",
                timeout=self.timeout
        ) as resp:
            result = Asset.model_validate(await resp.json())
            await queue.put(result)


class _VlmInferRequestAccepted(BaseModel):
    """Client facing API response model for accepted inference requests."""

    request_id: str = Field(
        description="Inference request Id, can be used to pull for updates"
    )


class _InferResponse(BaseModel):
    """Client-facing API response model matching WorkerResponse structure."""

    raw_output: str | None = Field(
        default=None, description="Raw text output from the model"
    )
    predictions: Sequence[Prediction] | None = Field(
        default=None, description="Structured predictions (eyepop format)"
    )
    run_info: InferRunInfo | None = Field(
        default=None, description="Runtime information about the inference execution"
    )

class InferJob(Job):
    timeout: aiohttp.ClientTimeout | None
    def __init__(
            self,
            asset_url: str,
            infer_request: InferRequest,
            session: ClientSession,
            on_ready: Callable[[DataJob], None] | None = None,
            callback: JobStateCallback | None = None,
            timeout: aiohttp.ClientTimeout | None = aiohttp.ClientTimeout(total=None, sock_read=60)
    ):
        super().__init__(session, on_ready, callback)
        self.timeout = timeout
        self._asset_url = asset_url
        self._infer_request = infer_request

        self._run_info = None

    @property
    def run_info(self) -> InferRunInfo | None:
        return self._run_info

    async def predict(self) -> dict[str, Any]:
        return await self.pop_result()

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        post_body_part = self._infer_request.model_dump(exclude_none=True)
        post_body_part["url"] = self._asset_url

        post_body = aiohttp.FormData()
        post_body.add_field('infer_request', json.dumps(post_body_part), content_type="application/json")

        total_timeout = self.timeout.total if self.timeout and self.timeout.total is not None is not None else 10.0 * 60.0
        start_time = time.time()
        request_id = None
        while time.time() - start_time < total_timeout:
            if request_id is None:
                request_coro = session.request_with_retry(
                    method="POST",
                    url="/api/v1/infer?timeout=20",
                    data=post_body,
                )
            else:
                request_coro = session.request_with_retry(
                    method="POST",
                    url=f"/api/v1/requests/{request_id}?timeout=20",
                )
            async with await request_coro as resp:
                if resp.status == 202:
                    request_id = _VlmInferRequestAccepted.model_validate(await resp.json()).request_id
                elif resp.status == 200:
                    result = _InferResponse.model_validate(await resp.json())
                    self._run_info = result.run_info = result.run_info
                    if result.predictions is not None:
                        for prediction in result.predictions:
                            await queue.put(prediction.model_dump(exclude_none=True))
                    break
                else:
                    raise ValueError(f"Unexpected status code: {resp.status}")


class EvaluateJob(Job):
    timeout: aiohttp.ClientTimeout | None
    def __init__(
            self,
            evaluate_request: EvaluateRequest,
            worker_release: str | None,
            session: ClientSession,
            on_ready: Callable[[DataJob], None] | None = None,
            callback: JobStateCallback | None = None,
            timeout: aiohttp.ClientTimeout | None = aiohttp.ClientTimeout(total=None, sock_read=60)
    ):
        super().__init__(session, on_ready, callback)
        self.timeout = timeout
        self._evaluate_request = evaluate_request
        self._worker_release = worker_release
        self._result = None

    @property
    async def response(self) -> EvaluateResponse | None:
        if not self._result:
            self._result = await self.pop_result()
        return self._result

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        worker_release_query = f'worker_release={self._worker_release}&' if self._worker_release is not None else ''
        total_timeout = self.timeout.total if self.timeout and self.timeout.total is not None is not None else 10.0 * 60.0
        start_time = time.time()
        request_id = None
        while time.time() - start_time < total_timeout:
            if request_id is None:
                request_coro = session.request_with_retry(
                    method="POST",
                    url=f"/api/v1/evaluations?timeout=20&{worker_release_query}",
                    content_type=APPLICATION_JSON,
                    data=self._evaluate_request.model_dump_json(exclude_none=True),
                )
            else:
                request_coro = session.request_with_retry(
                    method="GET",
                    url=f"/api/v1/evaluations/{request_id}?timeout=20",
                )
            async with await request_coro as resp:
                if resp.status == 202:
                    request_id = _VlmInferRequestAccepted.model_validate(await resp.json()).request_id
                elif resp.status == 200:
                    result = EvaluateResponse.model_validate(await resp.json())
                    await queue.put(result)
                    break
                else:
                    raise ValueError(f"Unexpected status code: {resp.status}")

