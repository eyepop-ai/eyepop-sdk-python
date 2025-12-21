import argparse
import asyncio
import logging
import os
from typing import Any, Callable
from uuid import uuid4

import aiohttp

from eyepop import EyePopSdk
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_jobs import DataJob
from eyepop.data.data_types import Asset, Dataset, DatasetCreate, AutoAnnotate, DatasetAutoAnnotate, \
    DatasetAutoAnnotateCreate, AutoAnnotateStatus, Prediction, PredictedClass, UserReview

logging.getLogger('eyepop').setLevel(logging.DEBUG)
logging.getLogger('eyepop.requests').setLevel(logging.DEBUG)


log = logging.getLogger(__name__)

class LocalAsset(Callable[[], Any]):
    def __init__(self, path: str) -> None:
        self.path = path
        self.file = None

    def fini(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.file is not None:
            self.file.close()
            self.file = None
        self.file = open(self.path, 'rb')
        return self.file


async def import_assets(endpoint: DataEndpoint, dataset: Dataset, assets_paths: str) -> list[Asset]:
    jobs = []
    assets = []
    async def on_ready(_job: DataJob, _local_asset: LocalAsset):
        nonlocal assets
        assets.append(await _job.result())
        _local_asset.fini()

    for asset_path in os.listdir(assets_paths):
        full_path = os.path.join(assets_paths, asset_path)
        if os.path.isfile(full_path):
            log.debug("importing asset: %s", asset_path)
            local_asset = LocalAsset(full_path)
            job = await endpoint.upload_asset_job(
                local_asset,
                dataset_uuid=dataset.uuid,
                mime_type='application/octet-stream',
                sync_transform=True,
                timeout=aiohttp.ClientTimeout(total=60*60)
            )
            jobs.append(on_ready(job, local_asset))
    await asyncio.gather(*jobs)
    log.debug("imported %d assets to %s", len(assets), dataset.uuid)
    return assets

async def auto_annotate_with_test_classes(
        endpoint: DataEndpoint,
        dataset: Dataset,
        auto_annotate: AutoAnnotate,
        source: str
) -> int:
    assets = await endpoint.list_assets(dataset_uuid=dataset.uuid)
    for asset in assets:
        await endpoint.add_asset_annotation(
            asset_uuid=asset.uuid,
            auto_annotate=auto_annotate,
            source=source,
            predictions=(Prediction(
                source_width=1.0,
                source_height=1.0,
                classes=[PredictedClass(
                    classLabel="test_class"
                )]
            ),),
        )
    return len(assets)

async def approve_auto_annotate(
        endpoint: DataEndpoint,
        dataset: Dataset,
        auto_annotate: AutoAnnotate,
        source: str
) -> int:
    assets = await endpoint.list_assets(dataset_uuid=dataset.uuid)
    for asset in assets:
        await endpoint.update_asset_annotation_approval(
            asset_uuid=asset.uuid,
            auto_annotate=auto_annotate,
            source=source,
            user_review=UserReview.approved,
        )
    return len(assets)

async def get_auto_annotate(
        endpoint: DataEndpoint,
        dataset: Dataset,
        auto_annotate: AutoAnnotate,
        source: str
) -> DatasetAutoAnnotate:
    pass

async def get_auto_annotate(
        endpoint: DataEndpoint,
        dataset: Dataset,
        auto_annotate: AutoAnnotate,
        source: str
) -> DatasetAutoAnnotate:
    pass

parser = argparse.ArgumentParser(
                    prog='import dataset',
                    description='Importing local assets into a dataset',
                    epilog='.')
parser.add_argument('assets_path',)
parser.add_argument('-r', '--remove', required=False, help="leave no trace, remove the dataset at the end", default=False, action="store_true")
parser.add_argument('-d', '--dataset-uuid', required=False, help="use this dataset uuid", default=None, type=str)
main_args = parser.parse_args()

async def main():
    async with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=10) as endpoint:
        try:
            if main_args.dataset_uuid is None:
                dataset = await endpoint.create_dataset(DatasetCreate(name="test import dataset"))
                log.debug("using newly created dataset: %s", dataset.uuid)
            else:
                dataset = await endpoint.get_dataset(main_args.dataset_uuid)
                log.debug("using existing dataset: %s", dataset.uuid)
            await import_assets(endpoint, dataset, main_args.assets_path)

            source = f"ep_evaluate:{uuid4().hex}"

            await endpoint.create_dataset_auto_annotate(
                auto_annotate_create=DatasetAutoAnnotateCreate(
                    auto_annotate="ep_evaluate",
                    source=source,
                    status=AutoAnnotateStatus.in_progress,
                ),
                dataset_uuid=dataset.uuid,
            )
            log.debug("auto annotate created for dataset: %s", dataset.uuid)
            num_assets = await auto_annotate_with_test_classes(
                endpoint=endpoint,
                dataset=dataset,
                auto_annotate="ep_evaluate",
                source=source,
            )
            log.debug("auto annotated %d assets in dataset: %s", num_assets, dataset.uuid)
            num_assets = await approve_auto_annotate(
                endpoint=endpoint,
                dataset=dataset,
                auto_annotate="ep_evaluate",
                source=source,
            )
            log.debug("approved %d asset annotations in dataset: %s", num_assets, dataset.uuid)
        finally:
            if main_args.dataset_uuid is None and main_args.remove:
                log.debug("removing dataset: %s", dataset.uuid)
                await endpoint.delete_dataset(dataset.uuid)

asyncio.run(main())
