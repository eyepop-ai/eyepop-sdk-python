import asyncio
import json
import logging
from importlib import resources

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

auto_annotate: str = "grounding_dino_base"
auto_annotate_params = AutoAnnotateParams(
    candidate_labels=["person", "car", "toy"]
)

async def create_dataset_and_train():
    async with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=1000, disable_ws=False) as endpoint:
        dataset = await endpoint.create_dataset(DatasetCreate(name="test_dataset"))
        try:
            log.info("created dataset: %s", dataset.uuid)
            await import_sample_assets(endpoint, dataset)
            await analyze_dataset(endpoint, dataset.uuid)
            await auto_annotate_dataset(endpoint, dataset.uuid, auto_annotate, auto_annotate_params)
            await approve_all(endpoint, dataset.uuid, dataset.modifiable_version, auto_annotate)
            model = await create_model(endpoint, dataset.uuid)
            model = await train_model(endpoint, model)
            log.info("model is trained and ready to us: %s", model.uuid)
        finally:
            await endpoint.delete_dataset(dataset.uuid)

async def import_sample_assets(endpoint: DataEndpoint, dataset: DatasetResponse) -> list[AssetResponse]:
    template_file = resources.files(sample_assets) / "sample_assets.json"
    with template_file.open("r") as f:
        sample_assets_json = json.load(f)
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
    analysis_future = asyncio.get_running_loop().create_future()
    async def dataset_event(event: ChangeEvent) -> None:
        if analysis_future.done():
            return
        if event.dataset_version is not None and event.change_type == ChangeType.dataset_version_modified:
            try:
                log.info("event: %s", event.model_dump_json())
                updated_dataset = await endpoint.get_dataset(event.dataset_uuid)
                log.info("event for dataset: %s", updated_dataset.model_dump_json())
                updated_version = next((v for v in updated_dataset.versions if v.version == event.dataset_version), None)
                log.info("event for dataset: %s", updated_version.model_dump_json() if updated_version else None)
                if updated_version is not None and updated_version.analysis_started_at is None:
                    analysis_future.set_result(None)
                    log.info("event: future done")
            except Exception as e:
                log.exception("event: future error", e)
                analysis_future.set_exception(e)

    await endpoint.add_dataset_event_handler(dataset_uuid, dataset_event)
    try:
        log.info("before analysis start for dataset %s", dataset_uuid)
        await endpoint.analyze_dataset_version(dataset_uuid)
        log.info("analysis started for dataset %s", dataset_uuid)
        await analysis_future
        log.info("analysis succeeded for dataset %s", dataset_uuid)
    finally:
        await endpoint.remove_dataset_event_handler(dataset_uuid, dataset_event)

async def auto_annotate_dataset(endpoint: DataEndpoint, dataset_uuid: str, auto_annotate_: str, auto_annotate_params_: AutoAnnotateParams) -> None:
    log.info("before auto annotate update dataset %s", dataset_uuid)
    await endpoint.update_dataset(dataset_uuid, DatasetUpdate(
        auto_annotates=[auto_annotate_],
        auto_annotate_params=auto_annotate_params_
    ), False)
    auto_annotate_future = asyncio.get_running_loop().create_future()
    async def dataset_event(event: ChangeEvent) -> None:
        if auto_annotate_future.done():
            return
        if event.dataset_version is not None and event.change_type == ChangeType.dataset_version_modified:
            try:
                updated_dataset = await endpoint.get_dataset(event.dataset_uuid)
                updated_version = next((v for v in updated_dataset.versions if v.version == event.dataset_version), None)
                if updated_version is not None and updated_version.auto_annotate_started_at is None:
                    auto_annotate_future.set_result(None)
            except Exception as e:
                auto_annotate_future.set_exception(e)

    await endpoint.add_dataset_event_handler(dataset_uuid, dataset_event)
    try:
        log.info("before auto annotates start for dataset %s", dataset_uuid)
        await endpoint.auto_annotate_dataset_version(dataset_uuid, None)
        log.info("auto annotates started for dataset %s", dataset_uuid)
        await auto_annotate_future
        log.info("auto annotates succeeded for dataset %s", dataset_uuid)
    finally:
        await endpoint.remove_dataset_event_handler(dataset_uuid, dataset_event)

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
    model = await endpoint.create_model(
        dataset.uuid,
        dataset_version.version,
        ModelCreate(name="sample model", description=""),
        False
    )
    log.info("model created: %s", model.model_dump_json())
    return model


async def train_model(endpoint: DataEndpoint, model: ModelResponse) -> ModelResponse:
    train_future = asyncio.get_running_loop().create_future()
    async def dataset_event(event: ChangeEvent) -> None:
        if train_future.done():
            return
        try:
            if event.change_type == ChangeType.model_progress:
                model_progress = await endpoint.get_model_progress(event.mdl_uuid)
                log.info("model training progress: %s", model_progress.model_dump_json())
            elif event.change_type == ChangeType.model_status_modified:
                changed_model = await endpoint.get_model(event.mdl_uuid)
                if changed_model.status == ModelStatus.available:
                    train_future.set_result(None)
                elif changed_model.status == ModelStatus.error:
                    train_future.set_exception(ValueError(changed_model.status_message))
        except Exception as e:
            train_future.set_exception(e)

    log.info("before training start for model %s", model.uuid)
    await endpoint.add_dataset_event_handler(model.dataset_uuid, dataset_event)
    try:
        model = await endpoint.train_model(model.uuid)
        log.info("training started for model %s", model.uuid)
        await train_future
        log.info("training succeeded for model %s", model.uuid)
        return model
    finally:
        await endpoint.remove_dataset_event_handler(model.dataset_uuid, dataset_event)

asyncio.run(create_dataset_and_train())

