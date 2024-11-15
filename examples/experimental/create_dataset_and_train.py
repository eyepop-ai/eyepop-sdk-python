import asyncio
import json
import logging
from importlib import resources

from examples.experimental.wait_for import WaitFor
from eyepop import EyePopSdk
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_jobs import DataJob
from eyepop.data.data_types import DatasetCreate, AssetImport, \
    AutoAnnotateParams, DatasetResponse, AssetResponse, ChangeEvent, ChangeType, DatasetUpdate, UserReview, \
    ModelResponse, ModelCreate, ModelStatus

from examples.experimental import sample_assets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

log = logging.getLogger(__name__)


async def import_sample_assets(endpoint: DataEndpoint, dataset: DatasetResponse) -> list[AssetResponse]:
    template_file = resources.files(sample_assets) / "sample_assets.json"
    with template_file.open("r") as f:
        sample_assets_json = json.load(f)
    sample_assets_json = sample_assets_json * 10
    log.info("before importing %d assets to %s", len(sample_assets_json), dataset.uuid)
    assets = []
    tasks = []
    async def on_ready(import_job: DataJob):
        assets.append(await import_job.result())
    for sample_asset in sample_assets_json:
        job = await endpoint.import_asset_job(AssetImport(
            url=sample_asset['asset_url'],
            ground_truth=sample_asset['prediction'],
        ), dataset.uuid)
        tasks.append(on_ready(job))
    await asyncio.gather(*tasks)
    log.info("imported %d assets to %s", len(assets), dataset.uuid)
    return assets

async def analyze_dataset(endpoint: DataEndpoint, dataset_uuid: str) -> None:
    log.info("before analysis start for dataset %s", dataset_uuid)
    await endpoint.analyze_dataset_version(dataset_uuid)
    log.info("analysis started for dataset %s", dataset_uuid)

async def auto_annotate_done_criteria(endpoint: DataEndpoint, event: ChangeEvent):
    if (event.dataset_version is not None and
            event.change_type in [ChangeType.dataset_version_modified, ChangeType.events_lost]):
        updated_dataset = await endpoint.get_dataset(event.dataset_uuid, include_stats=True)
        updated_version = next((v for v in updated_dataset.versions if v.version == event.dataset_version), None)
        return (updated_version is not None
                and updated_version.asset_stats.auto_annotated >= updated_version.asset_stats.accepted)

async def auto_annotate_dataset(endpoint: DataEndpoint, dataset_uuid: str,
                                auto_annotate: str, auto_annotate_params: AutoAnnotateParams) -> None:
    log.info("before auto annotate update dataset %s", dataset_uuid)
    await endpoint.update_dataset(dataset_uuid, DatasetUpdate(
        auto_annotates=[auto_annotate],
        auto_annotate_params=auto_annotate_params
    ), False)
    async with WaitFor(endpoint=endpoint, dataset_uuid=dataset_uuid, criteria=auto_annotate_done_criteria):
        log.info("before auto annotates start for dataset %s", dataset_uuid)
        await endpoint.auto_annotate_dataset_version(dataset_uuid, None)
        log.info("auto annotates started for dataset %s", dataset_uuid)
    log.info("auto annotates succeeded for dataset %s", dataset_uuid)

async def approve_all(endpoint: DataEndpoint, dataset_uuid: str, dataset_version: int, auto_annotate_: str) -> None:
    log.info("before approving all for dataset %s", dataset_uuid)
    assets = await endpoint.list_assets(dataset_uuid, dataset_version, False)
    for asset in assets:
        await endpoint.update_asset_auto_annotation_status(asset.uuid, auto_annotate_, UserReview.approved)
    log.info("%d auto annotations approved for dataset %s", len(assets), dataset_uuid)

async def create_model(endpoint: DataEndpoint, dataset_uuid: str) -> ModelResponse:
    log.info("before model creation for dataset: %s", dataset_uuid)
    dataset = await endpoint.freeze_dataset_version(dataset_uuid)
    log.info("dataset frozen: %s", dataset.model_dump_json())
    dataset_version = dataset.versions[1]
    model = await endpoint.create_model_from_dataset(
        dataset.uuid,
        dataset_version.version,
        ModelCreate(name="sample model", description=""),
        False
    )
    log.info("model created: %s", model.model_dump_json())
    return model

async def train_done_criteria(endpoint: DataEndpoint, event: ChangeEvent):
    if event.change_type == ChangeType.model_progress:
        model_progress = await endpoint.get_model_progress(event.mdl_uuid)
        log.info("model training progress: %s", model_progress.model_dump_json())
    elif event.change_type in [ChangeType.model_status_modified, ChangeType.events_lost]:
        changed_model = await endpoint.get_model(event.mdl_uuid)
        if changed_model.status == ModelStatus.available:
            return True
        elif changed_model.status == ModelStatus.error:
            raise ValueError(changed_model.status_message)

async def train_model(endpoint: DataEndpoint, model: ModelResponse) -> ModelResponse:
    async with WaitFor(endpoint=endpoint, dataset_uuid=model.dataset_uuid, criteria=train_done_criteria):
        log.info("before training start for model %s", model.uuid)
        model = await endpoint.train_model(model.uuid)
        log.info("training started for model %s", model.uuid)
    log.info("training succeeded for model %s", model.uuid)
    return model

async def main():
    auto_annotate: str = "grounding_dino_base"
    auto_annotate_params = AutoAnnotateParams(candidate_labels=["person"])

    async with EyePopSdk.dataEndpoint(is_async=True, disable_ws=False) as endpoint:
        dataset = await endpoint.create_dataset(DatasetCreate(name="test_dataset"))
        try:
            log.info("created dataset: %s", dataset.uuid)
            await import_sample_assets(endpoint, dataset)
            await analyze_dataset(endpoint, dataset.uuid)
            await auto_annotate_dataset(endpoint, dataset.uuid, auto_annotate, auto_annotate_params)
            await approve_all(endpoint, dataset.uuid, dataset.modifiable_version, auto_annotate)
            try:
                model = await create_model(endpoint, dataset.uuid)
                model = await train_model(endpoint, model)
                log.info("model is trained and ready to use: %s", model.uuid)
            finally:
                await endpoint.delete_model(model.uuid)
        finally:
            await endpoint.delete_dataset(dataset.uuid)

asyncio.run(main())
