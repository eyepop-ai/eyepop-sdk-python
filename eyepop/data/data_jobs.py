import json
from asyncio import Queue
from typing import Callable, BinaryIO

import aiohttp

from eyepop.client_session import ClientSession
from eyepop.data.data_types import AssetResponse, AssetImport
from eyepop.jobs import Job, JobStateCallback


class DataJob(Job):
    """
    Abstract Job submitted to an EyePop.ai DataEndpoint.
    """

    def __init__(self, session: ClientSession, on_ready: Callable[["DataJob"], None] | None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)

    async def result(self) -> AssetResponse:
        return await self.pop_result()

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        raise NotImplementedError("can't execute abstract jobs")


class _UploadStreamJob(DataJob):
    def __init__(self, stream: BinaryIO, mime_type: str, dataset_uuid: str, dataset_version: int | None,
                 external_id: str | None, session: ClientSession, on_ready: Callable[[DataJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self.stream = stream
        self.mime_type = mime_type
        self.dataset_uuid = dataset_uuid
        self.dataset_version = dataset_version
        self.external_id = external_id

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        dataset_version_query = f"&dataset_version={self.dataset_version}" if self.dataset_version else ""
        post_path: str
        if self.external_id:
            post_path = f"/assets?dataset_uuid={self.dataset_uuid}{dataset_version_query}&external_id={self.external_id}"
        else:
            post_path = f"/assets?dataset_uuid={self.dataset_uuid}{dataset_version_query}"

        async with await session.request_with_retry("POST", post_path, data=self.stream,
                                                    content_type=self.mime_type,
                                                    timeout=aiohttp.ClientTimeout(total=None, sock_read=60)) as resp:
            result = AssetResponse.model_validate(await resp.json())
            await queue.put(result)


class _ImportFromJob(DataJob):
    def __init__(self, asset_import: AssetImport, dataset_uuid: str, dataset_version: int | None,
                 external_id: str | None, partition: str | None,
                 session: ClientSession, on_ready: Callable[[DataJob], None] | None = None,
                 callback: JobStateCallback | None = None):
        super().__init__(session, on_ready, callback)
        self.asset_import = asset_import
        self.dataset_uuid = dataset_uuid
        self.dataset_version = dataset_version
        self.external_id = external_id
        self.partition = partition

    async def _do_execute_job(self, queue: Queue, session: ClientSession):
        dataset_version_query = f"&dataset_version={self.dataset_version}" if self.dataset_version else ""
        external_id_query = f"&external_id={self.external_id}" if self.external_id else ""
        partition_query = f"&partition={self.partition}" if self.partition else ""

        post_path = (f"/assets/imports?dataset_uuid={self.dataset_uuid}"
                     f"{dataset_version_query}{external_id_query}{partition_query}")

        async with await session.request_with_retry("POST", post_path, data=self.asset_import.model_dump_json(),
                                                    content_type="application/json",
                                                    timeout=aiohttp.ClientTimeout(total=None, sock_read=60)) as resp:
            result = AssetResponse.model_validate(await resp.json())
            await queue.put(result)
