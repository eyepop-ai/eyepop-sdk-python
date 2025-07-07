import asyncio
import json
from asyncio import StreamReader
from typing import Callable, BinaryIO, Any, AsyncIterable
from urllib.parse import urljoin, quote_plus

import aiohttp
import websockets
from pydantic import TypeAdapter
from websockets.asyncio.client import ClientConnection

from pydantic.tools import parse_obj_as

from eyepop.client_session import ClientSession
from eyepop.data.arrow.schema import MIME_TYPE_APACHE_ARROW_FILE_VERSIONED
from eyepop.data.data_jobs import DataJob, _UploadStreamJob, _ImportFromJob
from eyepop.data.data_syncify import SyncDataJob
from eyepop.data.data_types import Dataset, DatasetCreate, DatasetUpdate, Asset, Prediction, \
    AssetImport, AutoAnnotate, UserReview, TranscodeMode, Model, ModelCreate, ModelUpdate, \
    ModelTrainingProgress, ChangeEvent, ChangeType, EventHandler, ModelAlias, ModelAliasCreate, \
    ModelAliasUpdate, ModelExportFormat, QcAiHubExportParams, AssetUrlType, AssetInclusionMode, AnnotationInclusionMode, \
    ModelTrainingAuditRecord, ExportedUrlResponse, ModelTrainingEvent, ArtifactType
from eyepop.endpoint import Endpoint, log_requests

APPLICATION_JSON = "application/json"

WS_INITIAL_RECONNECT_DELAY = 1.0
WS_MAX_RECONNECT_DELAY = 60.0


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
    account_uuid: str
    data_config: dict[str, Any] | None

    disable_ws: bool
    ws: ClientConnection | None
    ws_tasks = set[asyncio.Task]
    ws_current_reconnect_delay: float | None
    account_event_handlers: set[EventHandler]
    dataset_uuid_to_event_handlers: dict[str, set[EventHandler]]

    def __init__(self, secret_key: str | None, access_token: str | None,
                 eyepop_url: str, account_id: str, job_queue_length: int,
                 request_tracer_max_buffer: int, disable_ws: bool = True):
        super().__init__(
            secret_key=secret_key, access_token=access_token, eyepop_url=eyepop_url,
            job_queue_length=job_queue_length, request_tracer_max_buffer=request_tracer_max_buffer
        )
        self.account_uuid = account_id
        self.data_config = None

        self.disable_ws = disable_ws
        self.ws = None
        self.ws_tasks = set()
        self.ws_current_reconnect_delay = None
        self.account_event_handlers = set()
        self.dataset_uuid_to_event_handlers = dict()

        self.add_retry_handler(404, self._retry_404)

    async def _retry_404(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('after 404, about to retry with fresh config')
            self.data_config = None
            return True

    async def _disconnect(self, timeout: float | None = None):
        await self._ws_disconnect()

    async def _reconnect(self):
        if self.data_config is not None:
            return
        config_url = f'{self.eyepop_url}/data/config?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("GET", config_url) as resp:
            self.data_config = await resp.json()
        if not self.disable_ws:
            await self._reconnect_ws()

    async def _reconnect_ws(self):
        await self._ws_disconnect()
        ws_url = urljoin(urljoin(
            self.eyepop_url, self.data_config['base_url']
        ).rstrip('/'),'events').replace(
            "https://", "wss://"
        ).replace(
            "http://", "ws://"
        )

        log_requests.debug("before ws connect: %s", ws_url)
        ws = await websockets.asyncio.client.connect(uri=ws_url)
        log_requests.debug("after ws connect: %s", ws_url)
        authorization_header = await self._authorization_header()
        if authorization_header is not None:
            auth_headers = {'authorization': authorization_header}
            message = json.dumps(auth_headers)
            await ws.send(message)
            log_requests.debug("ws send: %s", message)
        message = json.dumps({
            "subscribe" : {
                "account_uuid": self.account_uuid
            }
        })
        await ws.send(message)
        log_requests.debug("ws send: %s", message)
        for dataset_uuid in self.dataset_uuid_to_event_handlers:
            message = json.dumps({
                "subscribe" : {
                    "dataset_uuid": dataset_uuid
                }
            })
            await ws.send(message)
            log_requests.debug("ws send: %s", message)

        self.ws_current_reconnect_delay = WS_INITIAL_RECONNECT_DELAY
        ws_reader_task = asyncio.create_task(self._ws_reader(ws))

        self.ws = ws
        self.ws_tasks.add(ws_reader_task)
        ws_reader_task.add_done_callback(self.ws_tasks.discard)

    async def _ws_disconnect(self):
        ws = self.ws
        self.ws = None
        self.ws_reader_task = None
        if ws is not None:
            await ws.close()

    async def _ws_reader(self, ws: ClientConnection):
        try:
            async for message in ws:
                log_requests.debug("ws received %s", message)
                try:
                    data = json.loads(message)
                    if "change_type" in data:
                        change_event = ChangeEvent(**data)
                        await self._dispatch_change_event(change_event)
                except Exception as e:
                    log_requests.exception(e)
        except websockets.ConnectionClosed:
            log_requests.debug("ws disconnected")
            await self._ws_disconnect()
            if self.ws_current_reconnect_delay is None:
                self.ws_current_reconnect_delay = WS_INITIAL_RECONNECT_DELAY
            elif self.ws_current_reconnect_delay < WS_MAX_RECONNECT_DELAY:
                self.ws_current_reconnect_delay *= 1.5
            await asyncio.sleep(self.ws_current_reconnect_delay)
            ws_reconnect_task = asyncio.create_task(self._reconnect_ws())
            self.ws_tasks.add(ws_reconnect_task)
            ws_reconnect_task.add_done_callback(self.ws_tasks.discard)

    account_event_types = {
        ChangeType.dataset_added,
        ChangeType.dataset_modified,
        ChangeType.dataset_removed,
        ChangeType.dataset_version_modified
    }
    dataset_event_handlers = {
        ChangeType.dataset_added,
        ChangeType.dataset_modified,
        ChangeType.dataset_removed,
        ChangeType.dataset_version_modified,
        ChangeType.asset_added,
        ChangeType.asset_removed,
        ChangeType.asset_status_modified,
        ChangeType.asset_annotation_modified,
        ChangeType.model_added,
        ChangeType.model_modified,
        ChangeType.model_removed,
        ChangeType.model_status_modified,
        ChangeType.model_progress
    }

    async def _dispatch_change_event(self, change_event: ChangeEvent) -> None:
        if change_event.change_type in self.account_event_handlers:
            event_handlers = self.account_event_handlers.copy()
            for handler in event_handlers:
                await handler(change_event)
        if change_event.change_type in self.dataset_event_handlers:
            event_handlers = self.dataset_uuid_to_event_handlers.get(change_event.dataset_uuid, None)
            if event_handlers is not None:
                event_handlers = event_handlers.copy()
                for handler in event_handlers:
                    await handler(change_event)

    async def data_base_url(self) -> str:
        if self.data_config is None:
            await self._reconnect()
        return urljoin(self.eyepop_url, self.data_config['base_url']).rstrip("/")

    """ Event handlers """
    async def add_account_event_handler(self, event_handler: EventHandler):
        if self.disable_ws:
            raise ValueError("event handlers disabled, create endpoint with disable_ws=False "
                             "to register event handlers")
        self.account_event_handlers.add(event_handler)

    async def remove_account_event_handler(self, event_handler: EventHandler):
        self.account_event_handlers.discard(event_handler)

    async def add_dataset_event_handler(self, dataset_uuid: str, event_handler: EventHandler):
        if self.disable_ws:
            raise ValueError("event handlers disabled, create endpoint with disable_ws=False "
                             "to register event handlers")
        event_handlers = self.dataset_uuid_to_event_handlers.get(dataset_uuid, None)
        if event_handlers is None:
            event_handlers = set()
            self.dataset_uuid_to_event_handlers[dataset_uuid] = event_handlers
            ws = self.ws
            if ws:
                message = json.dumps({
                    "subscribe": {
                        "dataset_uuid": dataset_uuid
                    }
                })
                await ws.send(message)
                log_requests.debug("ws send: %s", message)

        event_handlers.add(event_handler)

    async def remove_dataset_event_handler(self, dataset_uuid: str, event_handler: EventHandler):
        event_handlers = self.dataset_uuid_to_event_handlers.get(dataset_uuid, None)
        if event_handlers is not None:
            event_handlers.discard(event_handler)
            if len(event_handlers) == 0:
                del self.dataset_uuid_to_event_handlers[dataset_uuid]
                ws = self.ws
                if ws:
                    message = json.dumps({
                        "unsubscribe": {
                            "dataset_uuid": dataset_uuid
                        }
                    })
                    await ws.send(message)
                    log_requests.debug("ws send: %s", message)

    async def remove_all_dataset_event_handlers(self, dataset_uuid: str):
        event_handlers = self.dataset_uuid_to_event_handlers.get(dataset_uuid, None)
        if event_handlers is not None:
            del self.dataset_uuid_to_event_handlers[dataset_uuid]
            ws = self.ws
            if ws:
                message = json.dumps({
                    "unsubscribe": {
                        "dataset_uuid": dataset_uuid
                    }
                })
                await ws.send(message)
                log_requests.debug("ws send: %s", message)

    """ Model methods """

    async def list_datasets(
            self,
            include_hero_asset: bool = False,
            modifiable_version_only: bool | None = None
    ) -> list[Dataset]:
        modifiable_version_only_query = f'&modifiable_version_only={modifiable_version_only}' if modifiable_version_only is not None else ''
        get_url = f'{await self.data_base_url()}/datasets?account_uuid={self.account_uuid}&include_hero_asset={include_hero_asset}{modifiable_version_only_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return TypeAdapter(list[Dataset]).validate_python(await resp.json())

    async def create_dataset(self, dataset: DatasetCreate) -> Dataset:
        post_url = f'{await self.data_base_url()}/datasets?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("POST", post_url, content_type=APPLICATION_JSON,
                                                 data=dataset.model_dump_json()) as resp:
            return TypeAdapter(Dataset).validate_python(await resp.json())

    async def get_dataset(
            self,
            dataset_uuid: str,
            dataset_version: int | None = None,
            include_stats: bool = False,
            modifiable_version_only: bool | None = None
    ) -> Dataset:
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        modifiable_version_only_query = f'&modifiable_version_only={modifiable_version_only}' if modifiable_version_only is not None else ''
        get_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}?include_stats={include_stats}{version_query}{modifiable_version_only_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return TypeAdapter(Dataset).validate_python(await resp.json())

    async def update_dataset(self, dataset_uuid: str, dataset: DatasetUpdate, start_auto_annotate: bool = True) -> Dataset:
        patch_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}?start_auto_annotate={start_auto_annotate}'
        log_requests.debug('update_dataset: %s', dataset.model_dump_json())
        async with await self.request_with_retry("PATCH", patch_url, content_type=APPLICATION_JSON,
                                                 data=dataset.model_dump_json(exclude_unset=True, exclude_none=True)) as resp:
            return TypeAdapter(Dataset).validate_python(await resp.json())

    async def delete_dataset(self, dataset_uuid: str) -> None:
        delete_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}'
        async with await self.request_with_retry("DELETE", delete_url):
            return

    async def analyze_dataset_version(self, dataset_uuid: str, dataset_version: int | None = None) -> None:
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        post_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}/analyze?{version_query}'
        async with await self.request_with_retry("POST", post_url):
            return

    async def auto_annotate_dataset_version(self, dataset_uuid: str, dataset_version: int | None = None, max_assets: int | None = None) -> None:
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        max_assets_query = f'&max_assets={max_assets}' if max_assets is not None else ''
        post_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}/auto_annotate?{version_query}{max_assets_query}'
        async with await self.request_with_retry("POST", post_url):
            return

    async def freeze_dataset_version(self, dataset_uuid: str, dataset_version: int | None = None) -> Dataset:
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        post_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}/freeze?{version_query}'
        async with await self.request_with_retry("POST", post_url) as resp:
            return parse_obj_as(Dataset, await resp.json())

    async def delete_dataset_version(self, dataset_uuid: str, dataset_version: int) -> Dataset:
        delete_url = f'{await self.data_base_url()}/datasets/{dataset_uuid}/versions?dataset_version={dataset_version}'
        async with await self.request_with_retry("DELETE", delete_url) as resp:
            return parse_obj_as(Dataset, await resp.json())

    async def delete_annotations(self, dataset_uuid: str, dataset_version: int,
                                 user_reviews: list[UserReview] = (UserReview.unknown,)) -> None:
        user_reviews_query = ""
        for user_review in user_reviews:
            user_reviews_query += f"&user_review={user_review}"
        delete_url = (f'{await self.data_base_url()}/datasets/{dataset_uuid}/annotations'
                    f'?dataset_version={dataset_version}{user_reviews_query}')
        async with await self.request_with_retry("DELETE", delete_url):
            return

    """" Asset methods """

    async def upload_asset_job(self, stream: BinaryIO, mime_type: str, dataset_uuid: str,
                               dataset_version: int | None = None, external_id: str | None = None,
                               on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        session = DataClientSession(self, await self.data_base_url())
        job = _UploadStreamJob(stream=stream, mime_type=mime_type, dataset_uuid=dataset_uuid,
                               dataset_version=dataset_version, external_id=external_id, session=session,
                               on_ready=on_ready, callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    async def import_asset_job(self, asset_import: AssetImport, dataset_uuid: str, dataset_version: int | None = None,
                               external_id: str | None = None, partition: str | None = None,
                               on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        session = DataClientSession(self, await self.data_base_url())
        job = _ImportFromJob(asset_import=asset_import, dataset_uuid=dataset_uuid, dataset_version=dataset_version,
                             external_id=external_id, session=session, partition=partition,
                             on_ready=on_ready, callback=self.metrics_collector)
        await self._task_start(job.execute())
        return job

    async def list_assets(self, dataset_uuid: str, dataset_version: int | None = None,
                          include_annotations: bool = False) -> list[Asset]:
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        get_url = f'{await self.data_base_url()}/assets?dataset_uuid={dataset_uuid}&include_annotations={"true" if include_annotations else "false"}{version_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(list[Asset], await resp.json())

    async def get_asset(self, asset_uuid: str, dataset_uuid: str | None = None,
                        dataset_version: int | None = None, include_annotations: bool = False) -> Asset:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        get_url = f'{await self.data_base_url()}/assets/{asset_uuid}?include_annotations={"true" if include_annotations else "false"}{dataset_query}{version_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(Asset, await resp.json())

    async def delete_asset(self, asset_uuid: str, dataset_uuid: str | None = None,
                           dataset_version: int | None = None) -> None:
        dataset_query = f'dataset_uuid={dataset_uuid}&' if dataset_uuid is not None else ''
        version_query = f'dataset_version={dataset_version}' if dataset_version is not None else ''
        delete_url = f'{await self.data_base_url()}/assets/{asset_uuid}?{dataset_query}{version_query}'
        async with await self.request_with_retry("DELETE", delete_url):
            return

    async def resurrect_asset(self, asset_uuid: str, dataset_uuid: str, from_dataset_version: int,
                              into_dataset_version: int | None = None) -> None:
        into_version_query = f'&into_dataset_version={into_dataset_version}' if into_dataset_version is not None else ''
        post_url = f'{await self.data_base_url()}/assets/{asset_uuid}/resurrect?dataset_uuid={dataset_uuid}&from_dataset_version={from_dataset_version}{into_version_query}'
        async with await self.request_with_retry("POST", post_url):
            return

    async def update_asset_ground_truth(self, asset_uuid: str, dataset_uuid: str | None = None,
                                        dataset_version: int | None = None,
                                        ground_truth: Prediction | None = None) -> None:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        patch_url = f'{await self.data_base_url()}/assets/{asset_uuid}/ground_truth?{dataset_query}{version_query}'
        async with await self.request_with_retry("PATCH", patch_url,
                                                 content_type=APPLICATION_JSON if ground_truth else None,
                                                 data=ground_truth.model_dump_json(
                                                     exclude_unset=True, exclude_none=True
                                                 ) if ground_truth else None):
            return

    async def delete_asset_ground_truth(self, asset_uuid: str, dataset_uuid: str | None = None,
                                        dataset_version: int | None = None) -> None:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        patch_url = f'{await self.data_base_url()}/assets/{asset_uuid}/ground_truth?{dataset_query}{version_query}'
        async with await self.request_with_retry("DELETE", patch_url):
            return

    async def update_asset_auto_annotation_status(self, asset_uuid: str, auto_annotate: AutoAnnotate,
                                                  user_review: UserReview, approved_threshold: float | None = None,
                                                  dataset_uuid: str | None = None,
                                                  dataset_version: int | None = None) -> None:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        threshold_query = f'&approved_threshold={approved_threshold}' if approved_threshold is not None else ''
        patch_url = (f'{await self.data_base_url()}/assets/{asset_uuid}/auto_annotations/{auto_annotate}/user_review/'
                     f'{user_review}?{dataset_query}{version_query}{threshold_query}')
        async with await self.request_with_retry("PATCH", patch_url):
            return

    async def download_asset(self, asset_uuid: str, dataset_uuid: str | None = None,
                             dataset_version: int | None = None,
                             transcode_mode: TranscodeMode = TranscodeMode.original) -> StreamReader:
        dataset_query = f'&dataset_uuid={dataset_uuid}' if dataset_uuid is not None else ''
        version_query = f'&dataset_version={dataset_version}' if dataset_version is not None else ''
        get_url = f'{await self.data_base_url()}/assets/{asset_uuid}/download?transcode_mode={transcode_mode}{dataset_query}{version_query}'
        resp = await self.request_with_retry("GET", get_url)
        return resp.content

    """ Model methods """

    async def list_models(self) -> list[Model]:
        get_url = f'{await self.data_base_url()}/models?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(list[Model], await resp.json())

    async def create_model(self, model: ModelCreate) -> Model:
        post_url = f'{await self.data_base_url()}/models?account_uuid={self.account_uuid}&start_training=False'
        async with await self.request_with_retry("POST", post_url, content_type=APPLICATION_JSON,
                                                 data=model.model_dump_json()) as resp:
            return parse_obj_as(Model, await resp.json())

    async def upload_model_artifact(self, model_uuid: str, model_format: ModelExportFormat, artifact_name: str,
                                    stream: BinaryIO, mime_type: str = 'application/octet-stream') -> None:
        put_url = f'{await self.data_base_url()}/models/{model_uuid}/exports/eyepop/formats/{model_format}/artifacts/{artifact_name}'
        async with await self.request_with_retry("PUT", put_url, data=stream, content_type=mime_type,
                                                 timeout=aiohttp.ClientTimeout(total=None, sock_read=60)):
            return

    async def create_model_from_dataset(self, dataset_uuid: str, dataset_version: int | None, model: ModelCreate, start_training: bool = True) -> Model:
        post_url = f'{await self.data_base_url()}/models?dataset_uuid={dataset_uuid}&dataset_version={dataset_version}&start_training={start_training}'
        async with await self.request_with_retry("POST", post_url, content_type=APPLICATION_JSON,
                                                 data=model.model_dump_json()) as resp:
            return parse_obj_as(Model, await resp.json())

    async def get_model(self, model_uuid: str) -> Model:
        get_url = f'{await self.data_base_url()}/models/{model_uuid}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(Model, await resp.json())

    async def get_model_progress(self, model_uuid: str) -> ModelTrainingProgress:
        get_url = f'{await self.data_base_url()}/models/{model_uuid}/progress'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(ModelTrainingProgress, await resp.json())

    async def update_model(self, model_uuid: str, model: ModelUpdate) -> Model:
        patch_url = f'{await self.data_base_url()}/models/{model_uuid}'
        async with await self.request_with_retry("PATCH", patch_url, content_type=APPLICATION_JSON,
                                                 data=model.model_dump_json(
                                                     exclude_unset=True, exclude_none=True
                                                 )) as resp:
            return parse_obj_as(Model, await resp.json())

    async def delete_model(self, model_uuid: str) -> None:
        delete_url = f'{await self.data_base_url()}/models/{model_uuid}'
        async with await self.request_with_retry("DELETE", delete_url):
            return

    async def train_model(self, model_uuid: str) -> Model:
        post_url = f'{await self.data_base_url()}/models/{model_uuid}/train'
        async with await self.request_with_retry("POST", post_url) as resp:
            return parse_obj_as(Model, await resp.json())

    async def publish_model(self, model_uuid: str) -> Model:
        post_url = f'{await self.data_base_url()}/models/{model_uuid}/publish'
        async with await self.request_with_retry("POST", post_url) as resp:
            return parse_obj_as(Model, await resp.json())

    """ Model aliases methods """

    async def list_model_aliases(self) -> list[ModelAlias]:
        get_url = f'{await self.data_base_url()}/model_aliases?account_uuid={self.account_uuid}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(list[ModelAlias], await resp.json())

    async def create_model_alias(self, model_alias: ModelAliasCreate, dry_run: bool = False) -> ModelAlias:
        post_url = f'{await self.data_base_url()}/model_aliases?account_uuid={self.account_uuid}&dry_run={dry_run}'
        async with await self.request_with_retry("POST", post_url, content_type=APPLICATION_JSON,
                                                 data=model_alias.model_dump_json()) as resp:
            return parse_obj_as(ModelAlias, await resp.json())

    async def get_model_alias(self, name: str) -> ModelAlias:
        get_url = f'{await self.data_base_url()}/model_aliases/{name}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return parse_obj_as(ModelAlias, await resp.json())

    async def delete_model_alias(self, name: str) -> None:
        delete_url = f'{await self.data_base_url()}/model_aliases/{name}'
        async with await self.request_with_retry("DELETE", delete_url):
            return

    async def update_model_alias(self, name: str, model_alias: ModelAliasUpdate) -> ModelAlias:
        patch_url = f'{await self.data_base_url()}/model_aliases/{name}'
        async with await self.request_with_retry("PATCH", patch_url, content_type=APPLICATION_JSON,
                                                 data=model_alias.model_dump_json(
                                                     exclude_unset=True, exclude_none=True
                                                 )) as resp:
            return parse_obj_as(ModelAlias, await resp.json())

    async def set_model_alias_tag(self, name: str, tag: str, model_uuid: str) -> None:
        patch_url = f'{await self.data_base_url()}/model_aliases/{name}/{tag}?model_uuid={model_uuid}'
        async with await self.request_with_retry("PATCH", patch_url):
            return

    async def delete_model_alias_tag(self, name: str, tag: str) -> None:
        delete_url = f'{await self.data_base_url()}/model_aliases/{name}/{tag}'
        async with await self.request_with_retry("DELETE", delete_url):
            return

    """ Arrow im and export methods """

    async def export_assets(
            self,
            dataset_uuid: str | None = None,
            dataset_version: int | None = None,
            asset_uuids: list[str] | None = None,
            model_uuid: str | None = None,
            transcode_mode: TranscodeMode = TranscodeMode.image_original_size,
            asset_url_type: AssetUrlType | None = None,
            inclusion_mode: AssetInclusionMode = AssetInclusionMode.annotated_only,
            annotation_inclusion_mode: AnnotationInclusionMode = AnnotationInclusionMode.all,
            include_external_ids: bool = False,
            freeze_dataset_version: bool | None = None,
            include_partitions: list[str] | None = None,
            include_auto_annotates: list[AutoAnnotate] | None = None,
            include_sources: list[str] | None = None,
    ) -> StreamReader:
        asset_url_type_query = f'asset_url_type={asset_url_type}&' if asset_url_type is not None else ''
        dataset_uuid_query = f'dataset_uuid={dataset_uuid}&' if dataset_uuid is not None else ''
        dataset_version_query = f'dataset_version={dataset_version}&' if dataset_version is not None else ''
        asset_uuids_query = ""
        if asset_uuids is not None:
            for asset_uuid in asset_uuids:
                asset_uuids_query += f"asset_uuid={asset_uuid}&"
        model_uuid_query = f'model_uuid={model_uuid}&' if model_uuid is not None else ''
        freeze_dataset_version_query = f'freeze_dataset_version={freeze_dataset_version}&' if freeze_dataset_version is not None else ''
        include_partitions_query = ""
        if include_partitions is not None:
            for include_partition in include_partitions:
                include_partitions_query += f"include_partition={include_partition}&"
        include_auto_annotates_query = ""
        if include_auto_annotates is not None:
            for include_auto_annotate in include_auto_annotates:
                include_auto_annotates_query += f"include_auto_annotate={include_auto_annotate}&"
        include_sources_query = ""
        if include_sources is not None:
            for include_source in include_sources:
                include_sources_query += f"include_source={include_source}&"
        get_url = (f'{await self.data_base_url()}/exports/assets?'
                   f'{dataset_uuid_query}'
                   f'{dataset_version_query}'
                   f'{asset_uuids_query}'
                   f'{model_uuid_query}'
                   f'transcode_mode={transcode_mode}&'
                   f'inclusion_mode={inclusion_mode}&'
                   f'annotation_inclusion_mode={annotation_inclusion_mode}&'
                   f'include_external_ids={"true" if include_external_ids else "false"}&'
                   f'{asset_url_type_query}'
                   f'{freeze_dataset_version_query}'
                   f'{include_partitions_query}'
                   f'{include_auto_annotates_query}'
                   f'{include_sources_query}')

        resp = await self.request_with_retry(
            method="GET",
            url=get_url,
            accept=MIME_TYPE_APACHE_ARROW_FILE_VERSIONED,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=600)
        )
        return resp.content

    async def import_assets(
            self,
            arrow_stream: BinaryIO | AsyncIterable[bytes],
            dataset_uuid: str | None = None,
            dataset_version: int | None = None,
            model_uuid: str | None = None
    ) -> None:
        dataset_uuid_query = f'dataset_uuid={dataset_uuid}&' if dataset_uuid is not None else ''
        dataset_version_query = f'dataset_version={dataset_version}&' if dataset_version is not None else ''
        model_uuid_query = f'model_uuid={model_uuid}&' if model_uuid is not None else ''
        post_url = (f'{await self.data_base_url()}/imports/assets?'
                   f'{dataset_uuid_query}'
                   f'{dataset_version_query}'
                   f'{model_uuid_query}')
        await self.request_with_retry(
            "POST", post_url,
            data=arrow_stream,
            content_type=MIME_TYPE_APACHE_ARROW_FILE_VERSIONED,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=600)
        )

    async def model_training_audits(self, model_uuids: list[str]) -> list[ModelTrainingAuditRecord]:
        model_uuid_query = ""
        for model_uuid in model_uuids:
            model_uuid_query += f"model_uuid={model_uuid}&"
        get_url = f'{await self.data_base_url()}/exports/model_training_audits?{model_uuid_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return TypeAdapter(list[ModelTrainingAuditRecord]).validate_python(await resp.json())


    async def export_model_urls(self, model_uuids: list[str], model_formats: list[ModelExportFormat], device_name: str | None) -> list[ExportedUrlResponse]:
        model_uuid_query = ""
        for model_uuid in model_uuids:
            model_uuid_query += f"model_uuid={model_uuid}&"
        model_format_query = ""
        for model_format in model_formats:
            model_format_query += f"model_format={model_format}&"
        device_name_query = f"device_name={quote_plus(device_name)}&" if device_name is not None else ""
        get_url = f'{await self.data_base_url()}/exports/model_urls?{model_uuid_query}{model_format_query}{device_name_query}'
        async with await self.request_with_retry("GET", get_url) as resp:
            return TypeAdapter(list[ExportedUrlResponse]).validate_python(await resp.json())


    async def export_model_artifacts(self, model_uuids: list[str], model_formats: list[ModelExportFormat], device_name: str | None, artifact_type: ArtifactType | None) -> StreamReader:
        model_uuid_query = ""
        for model_uuid in model_uuids:
            model_uuid_query += f"model_uuid={model_uuid}&"
        model_format_query = ""
        for model_format in model_formats:
            model_format_query += f"model_format={model_format}&"
        device_name_query = f"device_name={quote_plus(device_name)}&" if device_name is not None else ""
        type_query = f"artifact_type={artifact_type}&" if artifact_type is not None else ""

        get_url = f'{await self.data_base_url()}/exports/model_artifacts?{model_uuid_query}{model_format_query}{device_name_query}{type_query}'
        resp = await self.request_with_retry("GET", get_url)
        return resp.content


    async def model_training_event(self, model_training_event: ModelTrainingEvent, model_uuid: str) -> None:
        post_url = f'{await self.data_base_url()}/models/{model_uuid}/events'
        await self.request_with_retry(
            "POST", post_url,
            data=model_training_event.model_dump_json(),
            content_type=APPLICATION_JSON,
        )

    async def qc_ai_hub_model_export(self, model_uuid: str, export_params: list[QcAiHubExportParams]) -> None:
        post_url = f'{await self.data_base_url()}/models/{model_uuid}/exports/qc_ai_hub'
        await self.request_with_retry(
            "POST", post_url,
            data=TypeAdapter(list[QcAiHubExportParams]).dump_json(export_params),
            content_type=APPLICATION_JSON,
        )
