import json
import logging
from importlib import resources
from typing import cast

from examples.experimental.wait_for_sync import WaitForSync
from eyepop import EyePopSdk
from eyepop.data.data_syncify import SyncDataEndpoint, SyncDataJob
from eyepop.data.data_types import DatasetCreate, AssetImport, \
    AutoAnnotateParams, DatasetResponse, AssetResponse, ChangeEvent, ChangeType, DatasetUpdate, UserReview, \
    ModelResponse, ModelCreate, ModelStatus

from examples.experimental import sample_assets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.INFO)

log = logging.getLogger(__name__)


def import_sample_assets(endpoint: SyncDataEndpoint, dataset: DatasetResponse) -> list[AssetResponse]:
    template_file = resources.files(sample_assets) / "sample_assets.json"
    with template_file.open("r") as f:
        sample_assets_json = json.load(f)
    log.info("before importing %d assets to %s", len(sample_assets_json), dataset.uuid)
    assets = []
    jobs: list[SyncDataJob] = []
    for sample_asset in sample_assets_json:
        job = endpoint.import_asset_job(AssetImport(
            url=sample_asset['asset_url'],
            ground_truth=sample_asset['prediction'],
        ), dataset.uuid)
        jobs.append(cast(SyncDataJob, job))
    for job in jobs:
        assets.append(job.result())
    log.info("imported %d assets to %s", len(assets), dataset.uuid)
    return assets

def analysis_done_criteria(endpoint: SyncDataEndpoint, event: ChangeEvent):
    if (event.dataset_version is not None and
            event.change_type in [ChangeType.dataset_version_modified, ChangeType.events_lost]):
        updated_dataset = endpoint.get_dataset(event.dataset_uuid)
        updated_version = next((v for v in updated_dataset.versions if v.version == event.dataset_version), None)
        return (updated_version is not None and updated_version.analysis_started_at is None)

def analyze_dataset(endpoint: SyncDataEndpoint, dataset_uuid: str) -> None:
    with WaitForSync(endpoint=endpoint, dataset_uuid=dataset_uuid, criteria=analysis_done_criteria):
        log.info("before analysis start for dataset %s", dataset_uuid)
        endpoint.analyze_dataset_version(dataset_uuid)
        log.info("analysis started for dataset %s", dataset_uuid)
    log.info("analysis succeeded for dataset %s", dataset_uuid)

def auto_annotate_done_criteria(endpoint: SyncDataEndpoint, event: ChangeEvent):
    if (event.dataset_version is not None and
            event.change_type in [ChangeType.dataset_version_modified, ChangeType.events_lost]):
        updated_dataset = endpoint.get_dataset(event.dataset_uuid)
        updated_version = next((v for v in updated_dataset.versions if v.version == event.dataset_version), None)
        return (updated_version is not None and updated_version.auto_annotate_started_at is None)

def auto_annotate_dataset(endpoint: SyncDataEndpoint, dataset_uuid: str, auto_annotate: str, auto_annotate_params: AutoAnnotateParams) -> None:
    log.info("before auto annotate update dataset %s", dataset_uuid)
    endpoint.update_dataset(dataset_uuid, DatasetUpdate(
        auto_annotates=[auto_annotate],
        auto_annotate_params=auto_annotate_params
    ), False)
    with WaitForSync(endpoint=endpoint, dataset_uuid=dataset_uuid, criteria=analysis_done_criteria):
        log.info("before auto annotates start for dataset %s", dataset_uuid)
        endpoint.auto_annotate_dataset_version(dataset_uuid, None)
        log.info("auto annotates started for dataset %s", dataset_uuid)
    log.info("auto annotates succeeded for dataset %s", dataset_uuid)

def approve_all(endpoint: SyncDataEndpoint, dataset_uuid: str, dataset_version: int, auto_annotate_: str) -> None:
    log.info("before approving all for dataset %s", dataset_uuid)
    assets = endpoint.list_assets(dataset_uuid, dataset_version, False)
    for asset in assets:
        endpoint.update_asset_auto_annotation_status(asset.uuid, auto_annotate_, UserReview.approved)
    log.info("%d auto annotations approved for dataset %s", len(assets), dataset_uuid)

def create_model(endpoint: SyncDataEndpoint, dataset_uuid: str) -> ModelResponse:
    log.info("before model creation for dataset: %s", dataset_uuid)
    dataset = endpoint.freeze_dataset_version(dataset_uuid)
    log.info("dataset frozen: %s", dataset.model_dump_json())
    dataset_version = dataset.versions[1]
    model = endpoint.create_model_from_dataset(
        dataset.uuid,
        dataset_version.version,
        ModelCreate(name="sample model", description=""),
        False
    )
    log.info("model created: %s", model.model_dump_json())
    return model

def train_done_criteria(endpoint: SyncDataEndpoint, event: ChangeEvent):
    if event.change_type == ChangeType.model_progress:
        model_progress = endpoint.get_model_progress(event.mdl_uuid)
        log.info("model training progress: %s", model_progress.model_dump_json())
    elif event.change_type in [ChangeType.model_status_modified, ChangeType.events_lost]:
        changed_model = endpoint.get_model(event.mdl_uuid)
        if changed_model.status == ModelStatus.available:
            return True
        elif changed_model.status == ModelStatus.error:
            raise ValueError(changed_model.status_message)

def train_model(endpoint: SyncDataEndpoint, model: ModelResponse) -> ModelResponse:
    with WaitForSync(endpoint=endpoint, dataset_uuid=model.dataset_uuid, criteria=train_done_criteria):
        log.info("before training start for model %s", model.uuid)
        model = endpoint.train_model(model.uuid)
        log.info("training started for model %s", model.uuid)
    log.info("training succeeded for model %s", model.uuid)
    return model

def main():
    auto_annotate: str = "grounding_dino_base"
    auto_annotate_params = AutoAnnotateParams(candidate_labels=["person", "car", "toy"])

    with EyePopSdk.dataEndpoint(is_async=False, disable_ws=False) as e:
        endpoint = cast(SyncDataEndpoint, e)
        dataset = endpoint.create_dataset(dataset=DatasetCreate(name="test_dataset"))
        try:
            log.info("created dataset: %s", dataset.uuid)
            import_sample_assets(endpoint, dataset)
            analyze_dataset(endpoint, dataset.uuid)
            auto_annotate_dataset(endpoint, dataset.uuid, auto_annotate, auto_annotate_params)
            approve_all(endpoint, dataset.uuid, dataset.modifiable_version, auto_annotate)
            model = create_model(endpoint, dataset.uuid)
            model = train_model(endpoint, model)
            log.info("model is trained and ready to use: %s", model.uuid)
        finally:
            endpoint.delete_dataset(dataset.uuid)

main()
