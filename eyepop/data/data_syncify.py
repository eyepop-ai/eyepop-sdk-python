import typing
from typing import BinaryIO, Callable, Optional, List

from eyepop.data.data_jobs import DataJob
from eyepop.data.data_types import AssetImport, DatasetResponse, DatasetCreate, DatasetUpdate, AssetResponse, \
    Prediction, AutoAnnotate, UserReview, TranscodeMode, ModelResponse, ModelCreate, ModelUpdate
from eyepop.syncify import run_coro_thread_save, SyncEndpoint, submit_coro_thread_save


class SyncDataJob:
    def __init__(self, job: DataJob, event_loop):
        self.job = job
        self.event_loop = event_loop

    def result(self) -> dict:
        result = run_coro_thread_save(self.event_loop, self.job.result())
        return result

    def cancel(self):
        run_coro_thread_save(self.event_loop, self.job.cancel())


class SyncDataEndpoint(SyncEndpoint):
    def __init__(self, endpoint: "DataEndpoint"):
        super().__init__(endpoint)

    """ Model methods """

    def list_datasets(self, include_hero_asset: bool = False) -> List[DatasetResponse]:
        return run_coro_thread_save(self.event_loop, self.endpoint.list_datasets(include_hero_asset))

    def create_dataset(self, dataset: DatasetCreate) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.create_dataset(dataset))

    def get_dataset(self, dataset_uuid: str, include_hero_asset: bool = False) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop, self.get_dataset(dataset_uuid, include_hero_asset))

    def update_dataset(self, dataset_uuid: str, dataset: DatasetUpdate) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop, self.update_dataset(dataset_uuid, dataset))

    def delete_dataset(self, dataset_uuid: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.delete_dataset(dataset_uuid))

    def freeze_dataset_version(self, dataset_uuid: str, dataset_version: Optional[int] = None) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.freeze_dataset_version(dataset_uuid, dataset_version))

    def delete_dataset_version(self, dataset_uuid: str, dataset_version: int) -> DatasetResponse:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.delete_dataset_version(dataset_uuid, dataset_version))

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
                         external_id: Optional[str] = None,
                         on_ready: Callable[[DataJob], None] | None = None) -> DataJob | SyncDataJob:
        if on_ready is not None:
            raise TypeError("'on_ready' callback not supported for sync endpoints. "
                            "Use 'EyePopSdk.dataEndpoint(is_async=True)` to create "
                            "an endpoint with callback support")
        job = run_coro_thread_save(self.event_loop,
                                   self.endpoint.import_asset_job(asset_import, dataset_uuid, dataset_version,
                                                                   external_id,
                                                                   None))
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

    def update_asset_manual_annotation(self, asset_uuid: str, dataset_uuid: Optional[str] = None,
                                       dataset_version: Optional[int] = None,
                                       manual_annotation: Optional[Prediction] = None) -> None:
        return run_coro_thread_save(self.event_loop,
                                    self.endpoint.update_asset_manual_annotation(asset_uuid, dataset_uuid,
                                                                                  dataset_version, manual_annotation))

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

    def create_model(self, dataset_uuid: str, dataset_version: int, model: ModelCreate) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.create_model(dataset_uuid, dataset_version, model))

    def get_model(self, model_uuid: str) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.get_model(model_uuid))

    def update_model(self, model_uuid: str, model: ModelUpdate) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.update_model(model_uuid, model))

    def delete_model(self, model_uuid: str) -> None:
        return run_coro_thread_save(self.event_loop, self.endpoint.delete_model(model_uuid))

    def publish_model(self, model_uuid: str) -> ModelResponse:
        return run_coro_thread_save(self.event_loop, self.endpoint.publish_model(model_uuid))
