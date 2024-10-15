import asyncio
import json
import logging
from io import StringIO
from typing import BinaryIO
from uuid import uuid4

import aiohttp

from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_types import ModelCreate, ModelExportFormat, ModelUpdate, ModelStatus, ModelAliasCreate
from eyepop.eyepopsdk import EyePopSdk
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import Pop, InferenceComponent, InferenceType
from importlib import resources
from examples.experimental import sample_assets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('eyepop.requests').setLevel(level=logging.DEBUG)

log = logging.getLogger(__name__)

LABELS_URL = 'https://s3.amazonaws.com/models.eyepop.ai/releases/yolov7/1.0.1/models/YOLOv7/COCO/Latest/COCO_Latest_labels.txt'
WEIGHTS_URL = 'https://s3.amazonaws.com/models.eyepop.ai/releases/yolov7/1.0.1/models/YOLOv7/COCO/Latest/TensorFlowLite/float32/yolov7_YOLOv7_COCO_Latest_TensorFlowLite_float32.tflite'
MODEL_JSON_URL = 'https://s3.amazonaws.com/models.eyepop.ai/releases/yolov7/1.0.1/models/YOLOv7/COCO/Latest/TensorFlowLite/float32/model.json'

ALIAS = f"{uuid4().hex}.example.eyepop.ai"
TAG = "latest"

async def import_artifacts(endpoint: DataEndpoint, model_uuid: str):
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        log.info("importing from: %s", LABELS_URL)
        async with session.get(LABELS_URL) as resp:
            await endpoint.upload_model_artifact(
                model_uuid,
                ModelExportFormat.TensorFlowLite,
                'labels.txt',
                resp.content
            )
        log.info("importing from: %s", WEIGHTS_URL)
        async with session.get(WEIGHTS_URL) as resp:
            await endpoint.upload_model_artifact(
                model_uuid,
                ModelExportFormat.TensorFlowLite,
                'yolov7_YOLOv7_COCO_Latest_TensorFlowLite_float32.tflite',
                resp.content
            )
        log.info("importing from: %s", MODEL_JSON_URL)
        async with session.get(MODEL_JSON_URL) as resp:
            model_json = json.loads(await resp.content.read())
            # patch the model.json because we upload the labels.txt in a different location
            model_json['assets'][1]['url'] = "labels.txt"
            await endpoint.upload_model_artifact(
                model_uuid,
                ModelExportFormat.TensorFlowLite,
                'yolov7_YOLOv7_COCO_Latest_TensorFlowLite_float32.tflite',
                json.dumps(model_json),
                mime_type='application/json'
            )

async def finish_model(endpoint: DataEndpoint, model_uuid: str):
    await endpoint.update_model(model_uuid, ModelUpdate(status=ModelStatus.available))
    log.info("made available: %s", model_uuid)

async def set_model_alias(endpoint: DataEndpoint, model_uuid: str, alias: str, tag: str):
    await endpoint.publish_model(model_uuid)
    await endpoint.create_model_alias(ModelAliasCreate(name=alias))
    log.info("created alias: %s", alias)
    await endpoint.set_model_alias_tag(alias, tag, model_uuid)
    log.info("set alais %s:%s to %s", alias, tag, model_uuid)

async def use_model(worker: WorkerEndpoint, model_uuid: str):
    await worker.set_pop(Pop(
        components=[InferenceComponent(
            inferenceTypes=[InferenceType.OBJECT_DETECTION],
            modelUuid=model_uuid
        )]
    ))
    example_file = resources.files(sample_assets) / "exampple.jpg"
    with example_file.open("rb") as f:
        job = await worker.upload_stream(f, "image/jpeg")
        result = job.predict()
        log.info("inference result: %s", result)

async def main():
    async with EyePopSdk.dataEndpoint(is_async=True, disable_ws=False) as endpoint:
        model = await endpoint.create_model(ModelCreate(
            name="test model",
            description=""
        ))
        try:
            log.info("created model: %s", model.uuid)
            await import_artifacts(endpoint, model.uuid)
            await finish_model(endpoint, model.uuid)
            try:
                await set_model_alias(endpoint, model.uuid, ALIAS, TAG)
                log.info("model ready to use: %s - starting a pop now", model.uuid)
                async with EyePopSdk.workerEndpoint(pop_id='transient', is_async=True) as worker:
                    await use_model(worker, model.uuid)
            finally:
                await endpoint.delete_model_alias_tag(ALIAS, TAG)
                await endpoint.delete_model_alias(ALIAS)
        finally:
            await endpoint.delete_model(model.uuid)

asyncio.run(main())
