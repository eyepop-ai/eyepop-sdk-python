import asyncio
import typing
from typing import BinaryIO, Callable, Optional, List

from eyepop.data.data_jobs import DataJob
from eyepop.data.data_types import AssetImport, DatasetResponse, DatasetCreate, DatasetUpdate, AssetResponse, \
    Prediction, AutoAnnotate, UserReview, TranscodeMode, ModelResponse, ModelCreate, ModelUpdate, ModelTrainingProgress, \
    ChangeEvent, EventHandler, ModelAliasResponse, ModelAliasCreate, ModelAliasUpdate, ModelExportFormat
from eyepop.syncify import run_coro_thread_save, SyncEndpoint, submit_coro_thread_save

SyncEventHandler = Callable[[ChangeEvent], None]


class SyncDataJob:
    def __init__(self, job: DataJob, event_loop):
        self.job = job
        self.event_loop = event_loop

    def result(self) -> AssetResponse:
        result = run_coro_thread_save(self.event_loop, self.job.result())
        return result

    def cancel(self):
        run_coro_thread_save(self.event_loop, self.job.cancel())


def wrap_event_handler(event_handler: SyncEventHandler) -> EventHandler:
    async def async_event_handler(event: ChangeEvent):
        await asyncio.to_thread(event_handler, event)
    return async_event_handler

async def null_event_handler(_: ChangeEvent):
    pass

class SyncDataEndpoint(SyncEndpoint):
    event_handlers: dict[SyncEventHandler, EventHandler]

    def __init__(self, endpoint: "DataEndpoint"):
        super().__init__(endpoint)
        self.event_handlers = {}

    """ Event handlers """
    def add_account_event_handler(self, event_handler: SyncEventHandler):
        async_event_handler = wrap_event_handler(event_handler)
        run_coro_thread_save(
            self.event_loop,
            self.endpoint.add_account_event_handler(async_event_handler)
        )
        self.event_handlers[event_handler] = async_event_handler


    def remove_account_event_handler(self, event_handler: SyncEventHandler):
        async_event_handler = self.event_handlers.pop(event_handler, null_event_handler)
        if async_event_handler is null_event_handler:
            return
        run_coro_thread_save(
            self.event_loop,
            self.endpoint.remove_account_event_handler(async_event_handler)
        )

    def add_dataset_event_handler(self, dataset_uuid: str, event_handler: SyncEventHandler):
        async_event_handler = wrap_event_handler(event_handler)
        run_coro_thread_save(
            self.event_loop,
            self.endpoint.add_dataset_event_handler(dataset_uuid, async_event_handler)
        )
        self.event_handlers[event_handler] = async_event_handler

    def remove_dataset_event_handler(self, dataset_uuid: str, event_handler: SyncEventHandler):
        async_event_handler = self.event_handlers.pop(event_handler, null_event_handler)
        if async_event_handler is null_event_handler:
            return
        run_coro_thread_save(
            self.event_loop,
            self.endpoint.remove_dataset_event_handler(dataset_uuid, async_event_handler)
        )

    def remove_all_dataset_event_handlers(self, dataset_uuid: str):
        run_coro_thread_save(
            self.event_loop,
            self.endpoint.remove_all_dataset_event_handlers(dataset_uuid)
        )

    """ Model methods """

    def list_datasets(self, include_hero_asset: bool = False) -> List[DatasetResponse]:
        return run_coro_thread_save(self.event_loop, self.endpoint.list_datasets(include_hero_asset))

    def create_dataset(self, dataset: DatasetCreate) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.create_dataset(dataset))

    def get_dataset(self, dataset_uuid: str, include_hero_asset: bool = False) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_dataset(dataset_uuid, include_hero_asset))

    def update_dataset(self, dataset_uuid: str, dataset: DatasetUpdate, start_auto_annotate: bool = True) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.update_dataset(dataset_uuid, dataset, start_auto_annotate))

    def delete_dataset(self, dataset_uuid: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.delete_dataset(dataset_uuid))

    def analyze_dataset_version(self, dataset_uuid: str, dataset_version: int | None = None) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.analyze_dataset_version(dataset_uuid, dataset_version))

    def auto_annotate_dataset_version(self, dataset_uuid: str, dataset_version: int | None = None, max_assets: int | None = None) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.auto_annotate_dataset_version(dataset_uuid, dataset_version, max_assets))

    def freeze_dataset_version(self, dataset_uuid: str, dataset_version: Optional[int] = None) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.freeze_dataset_version(dataset_uuid, dataset_version))

    def delete_dataset_version(self, dataset_uuid: str, dataset_version: int) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.delete_dataset_version(dataset_uuid, dataset_version))

    def delete_annotations(self, dataset_uuid: str, dataset_version: int,
                           user_reviews: list[UserReview] = (UserReview.unknown,)) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.delete_annotations(dataset_uuid, dataset_version, user_reviews))

    """" Asset methods """

    def upload_asset_job(self, stream: BinaryIO, mime_type: str, dataset_uuid: str,
                         dataset_version: int | None = None, external_id: str | None = None,
                         on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        if on_ready is not None:
            raise TypeError("'on_ready' callback not supported for sync endpoints. "
                            "Use 'EyePopSdk.dataEndpoint(is_async=True)` to create "
                            "an endpoint with callback support")
        job = run_coro_thread_save(self.event_loop,
                                   self.endpoint.upload_asset_job(stream, mime_type, dataset_uuid, dataset_version,
                                                                   external_id, None))
        return SyncDataJob(job, self.event_loop)

    def import_asset_job(self, asset_import: AssetImport, dataset_uuid: str, dataset_version: Optional[int] = None,
                         external_id: Optional[str] = None, partition: Optional[str] = None,
                         on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        if on_ready is not None:
            raise TypeError("'on_ready' callback not supported for sync endpoints. "
                            "Use 'EyePopSdk.dataEndpoint(is_async=True)` to create "
                            "an endpoint with callback support")
        job = run_coro_thread_save(self.event_loop,
                                   self.endpoint.import_asset_job(
                                       asset_import=asset_import,
                                       dataset_uuid=dataset_uuid,
                                       dataset_version=dataset_version,
                                       external_id=external_id,
                                       partition=partition,
                                       on_ready=None))
        return SyncDataJob(job, self.event_loop)

    def list_assets(self, dataset_uuid: str, dataset_version: Optional[int] = None,
                    include_annotations: bool = False) -> List[AssetResponse]:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.list_assets(dataset_uuid, dataset_version, include_annotations))

    def get_asset(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                  dataset_version: Optional[int] = None, include_annotations: bool = False) -> AssetResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_asset(asset_uuid, dataset_uuid, dataset_version,
                                                                             include_annotations))

    def delete_asset(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                     dataset_version: Optional[int] = None) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.delete_asset(asset_uuid, dataset_uuid, dataset_version))

    def resurrect_asset(self, asset_uuid: str, dataset_uuid: str, from_dataset_version: int,
                        into_dataset_version: Optional[int] = None) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.resurrect_asset(asset_uuid, dataset_uuid, from_dataset_version,
                                                                   into_dataset_version))

    def update_asset_ground_truth(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                                  dataset_version: Optional[int] = None,
                                  ground_truth: Optional[Prediction] = None) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.update_asset_ground_truth(asset_uuid, dataset_uuid,
                                                                            dataset_version, ground_truth))

    def delete_asset_ground_truth(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                                  dataset_version: Optional[int] = None) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.delete_asset_ground_truth(asset_uuid, dataset_uuid,
                                                                            dataset_version))

    def update_asset_auto_annotation_status(self, asset_uuid: str, auto_annotate: AutoAnnotate,
                                            user_review: UserReview, dataset_uuid: Optional[str] = None,
                                            dataset_version: Optional[int] = None) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.update_asset_auto_annotation_status(asset_uuid, auto_annotate,
                                                                                       user_review, dataset_uuid,
                                                                                       dataset_version))

    def download_asset(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                       dataset_version: Optional[int] = None,
                       transcode_mode: TranscodeMode = TranscodeMode.original) -> typing.BinaryIO:
        async_stream_reader = run_coro_thread_save(
            self.event_loop,
            self.endpoint.download_asset(asset_uuid, dataset_uuid, dataset_version, transcode_mode)
        )
        sync_io = self._async_reader_to_sync_binary_io(async_stream_reader)
        return sync_io

    """ Model methods """

    def list_models(self) -> List[ModelResponse]:
        return run_coro_thread_save(self.event_loop, self.endpoint.list_models())

    def create_model(self, model: ModelCreate) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.create_model(model))

    def upload_model_artifact(self, model_uuid: str, model_format: ModelExportFormat, artifact_name: str,
                                    stream: BinaryIO, mime_type: str = 'application/octet-stream') -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.upload_model_artifact(model_uuid, model_format, artifact_name, stream, mime_type))

    def create_model_from_dataset(self, dataset_uuid: str, dataset_version: int, model: ModelCreate, start_training: bool = True) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.create_model_from_dataset(dataset_uuid, dataset_version, model, start_training))

    def get_model(self, model_uuid: str) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_model(model_uuid))

    def get_model_progress(self, model_uuid: str) -> ModelTrainingProgress:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_model_progress(model_uuid))

    def update_model(self, model_uuid: str, model: ModelUpdate) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.update_model(model_uuid, model))

    def delete_model(self, model_uuid: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.delete_model(model_uuid))

    def train_model(self, model_uuid: str) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.train_model(model_uuid))

    def publish_model(self, model_uuid: str) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.publish_model(model_uuid))

    """ Model aliases methods """

    def list_model_aliases(self) -> list[ModelAliasResponse]:
        return run_coro_thread_save(self.event_loop, self.endpoint.list_model_aliases())

    def create_model_alias(self, model_alias: ModelAliasCreate, dry_run: bool = False) -> ModelAliasResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.create_model_alias(model_alias, dry_run))

    def get_model_alias(self, name: str) -> ModelAliasResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_model_alias(name))

    def delete_model_alias(self, name: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.delete_model_alias(name))

    def update_model_alias(self, name: str, model_alias: ModelAliasUpdate) -> ModelAliasResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.update_model_alias(name, model_alias))

    def set_model_alias_tag(self, name: str, tag: str, model_uuid: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.set_model_alias_tag(name, tag, model_uuid))

    def delete_model_alias_tag(self, name: str, tag: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.delete_model_alias_tag(name, tag))
