import asyncio
import logging
import os.path
import sys
import tempfile
from typing import cast

from eyepop import EyePopSdk
from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_types import DatasetCreate, Prediction, PredictedObject, AssetImport, ChangeEvent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

example_image_path = sys.argv[1]

async def create_dataset_and_train():
    with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=1000, disable_ws=False) as endpoint:
        async def event_handler(event: ChangeEvent):
            print(event)

        await endpoint.add_account_event_handler(event_handler)

        dataset = await endpoint.create_dataset(DatasetCreate(name="test_dataset"))
        try:
            print(dataset)
            asset_import = AssetImport(
                url="https://nmaahc.si.edu/sites/default/files/styles/max_1300x1300/public/images/header/audience-citizen_0.jpg",
                ground_truth=Prediction(source_width=1.0, source_height=1.0, objects=[
                    PredictedObject(classLabel="stuff", confidence=0.99, x=0.4, y=0.4, width=0.2, height=0.2)
                ])
            )
            jobs = []
            for i in range(100):
                jobs.append(endpoint.import_asset_job(asset_import=asset_import, dataset_uuid=dataset.uuid,
                                                      external_id=f"#{i}", partition="foo"))
            for job in jobs:
                asset = job.result()
                print(asset)

            assets = endpoint.list_assets(dataset_uuid=dataset.uuid)
            print(f"found {len(assets)} assets")

            asset = endpoint.get_asset(assets[0].uuid)
            print(asset)

            tmp_dir = tempfile.mkdtemp()
            tmp_file = os.path.join(tmp_dir, f"asset_{asset.uuid}.jpg")
            with open(tmp_file, "wb") as f:
                asset_stream = endpoint.download_asset(asset.uuid)
                f.write(asset_stream.read())
            print(f"downloaded asset {asset.uuid} to {tmp_file}")
        finally:
            endpoint.delete_dataset(dataset.uuid)

asyncio.run(create_dataset_and_train())

