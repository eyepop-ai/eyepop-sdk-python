import logging
import sys

from eyepop import EyePopSdk
from eyepop.data.data_types import DatasetCreate, Prediction, PredictedObject, AssetImport

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

example_image_path = sys.argv[1]

with (EyePopSdk.dataEndpoint(job_queue_length=1000) as endpoint):
    dataset = endpoint.create_dataset(DatasetCreate(name="test_dataset"))
    try:
        print(dataset)
        asset_import = AssetImport(
            url="https://nmaahc.si.edu/sites/default/files/styles/max_1300x1300/public/images/header/audience-citizen_0.jpg",
            manual_annotation=Prediction(source_width=1.0, source_height=1.0, objects=[
                PredictedObject(classLabel="stuff", confidence=0.99, x=0.4, y=0.4, width=0.2, height=0.2)
            ])
        )
        jobs = []
        for i in range(1000):
            jobs.append(endpoint.import_asset_job(asset_import=asset_import, dataset_uuid=dataset.uuid, external_id=f"#{i}"))
        for job in jobs:
            asset = job.result()
            print(asset)

    finally:
        endpoint.delete_dataset(dataset.uuid)



