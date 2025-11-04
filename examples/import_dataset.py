import argparse
import asyncio
import logging
import os
from typing import Any, Callable

import aiohttp

from eyepop import EyePopSdk
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_jobs import DataJob
from eyepop.data.data_types import DatasetCreate, Dataset, Asset


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

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
            log.info("importing asset: %s", asset_path)
            local_asset = LocalAsset(full_path)
            job = await endpoint.upload_asset_job(
                local_asset,
                dataset_uuid=dataset.uuid,
                mime_type='application/octet-stream',
                sync_transform=True,
                timeout=aiohttp.ClientTimeout(total=10*60)
            )
            jobs.append(on_ready(job, local_asset))
    await asyncio.gather(*jobs)
    log.info("imported %d assets to %s", len(assets), dataset.uuid)
    return assets


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
                log.info("using newly created dataset: %s", dataset.uuid)
            else:
                dataset = await endpoint.get_dataset(main_args.dataset_uuid)
                log.info("using existing dataset: %s", dataset.uuid)
            await import_assets(endpoint, dataset, main_args.assets_path)
        finally:
            if main_args.dataset_uuid is None and main_args.remove:
                log.info("removing dataset: %s", dataset.uuid)
                await endpoint.delete_dataset(dataset.uuid)

asyncio.run(main())
