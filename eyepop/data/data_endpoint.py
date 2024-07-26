from asyncio import StreamReader
from typing import Callable, Any, BinaryIO, List, Optional
from urllib.parse import urljoin

import aiohttp
from pydantic.tools import parse_obj_as

from eyepop.client_session import ClientSession
from eyepop.data.data_jobs import DataJob, _UploadStreamJob, _ImportFromJob
from eyepop.data.data_syncify import SyncDataJob
from eyepop.data.data_types import DatasetResponse, DatasetCreate, DatasetUpdate, AssetResponse, Prediction, \
    AssetImport, AutoAnnotate, UserReview, TranscodeMode, ModelResponse, ModelCreate, ModelUpdate
from eyepop.endpoint import Endpoint, log_requests

APPLICATION_JSON = "application/json"


class DataClientSession(ClientSession):
    def __init__(self, delegee: ClientSession, base_url: str):
        self.delegee = delegee
        self.base_url = base_url

    async def request_with_retry(self, method: str, url: str, accept: str | None = None, data: Any = None,
                                 content_type: str | None = None,
                                 timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        url = urljoin(self.base_url, url)
        return await self.delegee.request_with_retry(method, url, accept, data, content_type, timeout)


class DataEndpoint(Endpoint):
    """
    Endpoint to the EyePop.ai Data API.
    """

    def __init__(self, secret_key: str, eyepop_url: str, account_id: str, job_queue_length: int,
                 request_tracer_max_buffer: int):
        super().__init__(secret_key, eyepop_url, job_queue_length, request_tracer_max_buffer)
        self.account_uuid = account_id
        self.data_config = None
        self.add_retry_handler(404, self._retry_404)

    async def _retry_404(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('after 404, about to retry with fresh config')
            self.data_config = None
            return True

    async def _disconnect(self):
        pass

    async def _reconnect(self):
        if self.data_config is not None:
            return
        config_url = f'{self.eyepop_url}/data/config?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("GET", config_url) as resp:
            self.data_config = await resp.json()

    async def data_base_url(self) -> str:
        if self.data_config is None:
            await self._reconnect()
        return urljoin(self.eyepop_url, self.data_config['base_url']).rstrip("/")

    """ Model methods """

    async def list_datasets(self, include_hero_asset: bool = False) -> List[DatasetResponse]:
        get_url = f'{await self.data_base_url()}/datasets?account_uuid={self.account_uuid}&include_hero_asset={include_hero_asset}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(List[DatasetResponse], await resp.json())

    async def create_dataset(self, dataset: DatasetCreate) -> DatasetResponse:
        post_url = f'{await self.data_base_url()}/datasets?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("POST", post_url, content_type=APPLICATION_JSON,
                                                 data=dataset.json()) as resp:
            return parse_obj_as(DatasetResponse, await resp.json())

    async def get_dataset(self, dataset_uuid: str, include_hero_asset: bool = False) -> DatasetResponse:
        get_url = f'{await self.data_base_url()}/datasets?dataset_uuid={dataset_uuid}&include_hero_asset={include_hero_asset}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(DatasetResponse, await resp.json())

    async def update_dataset(self, dataset_uuid: str, dataset: DatasetUpdate) -> DatasetResponse:
        patch_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}'
        async with await self.request_with_retry("PATCH", patch_url, content_type=APPLICATION_JSON,
                                                 data=dataset.json()) as resp:
            return parse_obj_as(DatasetResponse, await resp.json())

    async def delete_dataset(self, dataset_uuid: str) -> None:
        delete_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}'
        async with await self.request_with_retry("DELETE", delete_url) as resp:
            return

    async def freeze_dataset_version(self, dataset_uuid: str, dataset_version: Optional[int] = None) -> DatasetResponse:
        if dataset_version:
            post_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}?dataset_version={dataset_version}'
        else:
            post_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}'
        async with await self.request_with_retry("POST", post_url) as resp:
            return parse_obj_as(DatasetResponse, await resp.json())

    async def delete_dataset_version(self, dataset_uuid: str, dataset_version: int) -> DatasetResponse:
        post_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}/delete?dataset_version={dataset_version}'
        async with await self.request_with_retry("POST", post_url) as resp:
            return parse_obj_as(DatasetResponse, await resp.json())

    """" Asset methods """

    async def upload_asset_job(self, stream: BinaryIO, mime_type: str, dataset_uuid: str,
                               dataset_version: Optional[int] = None, external_id: Optional[str] = None,
                               on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        session = DataClientSession(self, await self.data_base_url())
        job = _UploadStreamJob(stream=stream, mime_type=mime_type, dataset_uuid=dataset_uuid,
                               dataset_version=dataset_version, external_id=external_id, session=session,
                               on_ready=on_ready, callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    async def import_asset_job(self, asset_import: AssetImport, dataset_uuid: str, dataset_version: Optional[int] = None,
                               external_id: Optional[str] = None,
                               on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        session = DataClientSession(self, await self.data_base_url())
        job = _ImportFromJob(asset_import=asset_import, dataset_uuid=dataset_uuid, dataset_version=dataset_version,
                             external_id=external_id,session=session,
                             on_ready=on_ready, callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    async def list_assets(self, dataset_uuid: str, dataset_version: Optional[int] = None,
                          include_annotations: bool = False) -> List[AssetResponse]:
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        get_url = f'{await self.data_base_url()}/assets?dataset_uuid={dataset_uuid}&include_annotations={"true" if include_annotations else "false"}{version_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(List[AssetResponse], await resp.json())

    async def get_asset(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                        dataset_version: Optional[int] = None, include_annotations: bool = False) -> AssetResponse:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        get_url = f'{await self.data_base_url()}/assets/{asset_uuid}?include_annotations={"true" if include_annotations else "false"}{dataset_query}{version_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(AssetResponse, await resp.json())

    async def delete_asset(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                           dataset_version: Optional[int] = None) -> None:
        dataset_query = f'dataset_uuid={dataset_uuid}&' if dataset_uuid is not None else ''
        version_query = f'dataset_version={dataset_version}' if dataset_version is not None else ''
        delete_url = f'{await self.data_base_url()}/assets/{asset_uuid}?{dataset_query}{version_query}'
        async with await self.request_with_retry("DELETE", delete_url) as resp:
            return

    async def resurrect_asset(self, asset_uuid: str, dataset_uuid: str, from_dataset_version: int,
                              into_dataset_version: Optional[int] = None) -> None:
        into_version_query = f'&into_dataset_version={into_dataset_version}' if into_dataset_version is not None else ''
        post_url = f'{await self.data_base_url()}/assets/{asset_uuid}/resurrect?dataset_uuid={dataset_uuid}&from_dataset_version={from_dataset_version}{into_version_query}'
        async with await self.request_with_retry("POST", post_url) as resp:
            return

    async def update_asset_manual_annotation(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                                             dataset_version: Optional[int] = None,
                                             manual_annotation: Optional[Prediction] = None) -> None:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        patch_url = f'{await self.data_base_url()}/assets/{asset_uuid}/manual_annotate?{dataset_query}{version_query}'
        async with await self.request_with_retry("PATCH", patch_url,
                                                 content_type=APPLICATION_JSON if manual_annotation else None,
                                                 data=manual_annotation.json() if manual_annotation else None) as resp:
            return

    async def update_asset_auto_annotation_status(self, asset_uuid: str, auto_annotate: AutoAnnotate,
                                                  user_review: UserReview, dataset_uuid: Optional[str] = None,
                                                  dataset_version: Optional[int] = None) -> None:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        patch_url = f'{await self.data_base_url()}/assets/{asset_uuid}/auto_annotations/{auto_annotate}/user_review/{user_review}?{dataset_query}{version_query}'
        async with await self.request_with_retry("PATCH", patch_url) as resp:
            return

    async def download_asset(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                             dataset_version: Optional[int] = None,
                             transcode_mode: TranscodeMode = TranscodeMode.original) -> StreamReader:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        get_url = f'{await self.data_base_url()}/assets/{asset_uuid}/download?transcode_mode={transcode_mode}{dataset_query}{version_query}'
        resp = await self.request_with_retry("GET", get_url)
        return resp.content

    """ Model methods """

    async def list_models(self) -> List[ModelResponse]:
        get_url = f'{await self.data_base_url()}/models?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(List[ModelResponse], await resp.json())

    async def create_model(self, dataset_uuid: str, dataset_version: int, model: ModelCreate) -> ModelResponse:
        post_url = f'{await self.data_base_url()}/models?dataset_uuid={dataset_uuid}&dataset_version={dataset_version}'
        async with await self.request_with_retry("POST", post_url, content_type=APPLICATION_JSON,
                                                 data=model.json()) as resp:
            return parse_obj_as(ModelResponse, await resp.json())

    async def get_model(self, model_uuid: str) -> ModelResponse:
        get_url = f'{await self.data_base_url()}/models/{model_uuid}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(ModelResponse, await resp.json())

    async def update_model(self, model_uuid: str, model: ModelUpdate) -> ModelResponse:
        patch_url = f'{await self.data_base_url()}/models/{model_uuid}'
        async with await self.request_with_retry("PATCH", patch_url, content_type=APPLICATION_JSON,
                                                 data=model.json()) as resp:
            return parse_obj_as(ModelResponse, await resp.json())

    async def delete_model(self, model_uuid: str) -> None:
        delete_url = f'{await self.data_base_url()}/models/{model_uuid}'
        async with await self.request_with_retry("DELETE", delete_url) as resp:
            return

    async def publish_model(self, model_uuid: str) -> ModelResponse:
        post_url = f'{await self.data_base_url()}/models/{model_uuid}/publish'
        async with await self.request_with_retry("POST", post_url) as resp:
            return parse_obj_as(ModelResponse, await resp.json())
