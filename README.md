# EyePop.ai Python SDK

Python SDK for EyePop.ai's inference and data APIs.

- **Install**: `pip install eyepop`
- **Requires**: Python 3.12+
- **Examples**: see [`examples/`](examples/)

## Quickstart

```python
from eyepop import EyePopSdk

with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.upload('examples/example.jpg').predict()
    print(result)
```

Set `EYEPOP_API_KEY` in your environment (get one at [dashboard.eyepop.ai](https://dashboard.eyepop.ai)), or pass `api_key=...` to `sync_worker()`.

## Configuration

Credentials are read from environment variables. Set **one** auth method:

| Variable | Description |
|---|---|
| `EYEPOP_API_KEY` | API key from your dashboard. |
| `EYEPOP_ACCESS_TOKEN` | Pre-issued OAuth access token. |

Optional:

| Variable | Description |
|---|---|
| `EYEPOP_POP_ID` | Named pop ID. Defaults to `transient`. |
| `EYEPOP_SESSION_UUID` | Reuse an existing worker session. |
| `EYEPOP_URL` | Override API endpoint. |
| `EYEPOP_LOCAL_MODE` | Connect to a worker at `127.0.0.1:8080`. |
| `EYEPOP_PIPELINE_IMAGE` | Custom worker Docker image. |
| `EYEPOP_PIPELINE_VERSION` | Custom worker image tag. |
| `EYEPOP_ACCOUNT_ID` | Required for some Data API calls. |

Use [python-dotenv](https://pypi.org/project/python-dotenv/) to load a `.env` file (see [`.env.example`](.env.example)).

You can also pass credentials explicitly:

```python
endpoint = EyePopSdk.sync_worker(pop_id='my-pop-id', api_key='my-api-key')
```

## Usage

### Single image

```python
from eyepop import EyePopSdk

with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.upload('examples/example.jpg').predict()
    print(result)
```

`upload()` queues the file; `predict()` blocks until the result is ready. For videos or multi-frame containers, call `predict()` in a loop until it returns `None`.

### Binary streams

```python
with EyePopSdk.sync_worker() as endpoint:
    with open('examples/example.jpg', 'rb') as file:
        result = endpoint.upload_stream(file, 'image/jpeg').predict()
```

### URLs (HTTP, RTSP, RTMP)

```python
with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.load_from('https://example.com/image.jpg').predict()
```

### Videos

```python
with EyePopSdk.sync_worker() as endpoint:
    job = endpoint.load_from('https://example.com/video.mp4')
    while result := job.predict():
        print(result)
```

Cancel a job mid-stream with `job.cancel()`.

### Batching

Queue multiple uploads, then collect results:

```python
with EyePopSdk.sync_worker() as endpoint:
    jobs = [endpoint.upload(p) for p in file_paths]
    for job in jobs:
        print(job.predict())
```

### Async with callbacks

```python
import asyncio
from eyepop import EyePopSdk, Job

async def main(paths):
    async def on_ready(job: Job):
        print(await job.predict())

    async with EyePopSdk.async_worker() as endpoint:
        for p in paths:
            await endpoint.upload(p, on_ready=on_ready)

asyncio.run(main(['examples/example.jpg'] * 100))
```

### Visualize results

```python
from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk

with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.upload('examples/example.jpg').predict()

with Image.open('examples/example.jpg') as image:
    plt.imshow(image)
EyePopSdk.plot(plt.gca()).prediction(result)
plt.show()
```

See [`examples/visualize_with_webui2.py`](examples/visualize_with_webui2.py) for an interactive viewer.

## Composable Pops

Build multi-stage inference pipelines by chaining models. Configure at runtime with `endpoint.set_pop(pop)`.

### Components

| Component | Purpose |
|---|---|
| `InferenceComponent` | Run a model. Supports chunked video via `videoChunkLengthSeconds` / `videoChunkOverlap`. |
| `TrackingComponent` | Track detected objects across frames. |
| `ContourFinderComponent` | Extract contours from segmentation masks. |
| `ComponentFinderComponent` | Extract connected components from masks. |
| `ForwardComponent` | Route outputs between stages. |

### Forwarding

- **`CropForward`** — pass each detection crop to sub-components.
- **`FullForward`** — pass the full image to sub-components.

Both accept `includeClasses` to filter forwarded detections.

### Example: Vehicle → License Plate → OCR

```python
from eyepop.worker.worker_types import (
    Pop, InferenceComponent, TrackingComponent, CropForward, MotionModel,
)

pop = Pop(components=[
    InferenceComponent(
        ability='eyepop.vehicle:latest',
        categoryName='vehicles',
        confidenceThreshold=0.8,
        forward=CropForward(targets=[
            TrackingComponent(
                maxAgeSeconds=5.0,
                motionModel=MotionModel.CONSTANT_VELOCITY,
                agnostic=True,
            ),
            InferenceComponent(
                ability='eyepop.vehicle.licence-plate:latest',
                topK=1,
                forward=CropForward(targets=[
                    InferenceComponent(
                        ability='eyepop.text.recognize.landscape:latest',
                        categoryName='licence-plate',
                    ),
                ]),
            ),
        ]),
    ),
])
```

### Example: VLM open-vocabulary detection

```python
from eyepop.worker.worker_types import Pop, InferenceComponent, CropForward

pop = Pop(components=[
    InferenceComponent(
        ability='eyepop.localize-objects:latest',
        params={'prompts': [{'prompt': 'person'}]},
        forward=CropForward(targets=[
            InferenceComponent(
                ability='eyepop.image-contents:latest',
                params={'prompts': [{'prompt': 'hair color?'}]},
            ),
        ]),
    ),
])
```

Full catalogue: face mesh, hand tracking, body pose, SAM segmentation, and more in [`examples/pop_demo.py`](examples/pop_demo.py).

## Data Endpoint

Dataset management, VLM inference, and evaluation workflows.

```python
import asyncio
from eyepop import EyePopSdk

async def main():
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        datasets = await endpoint.list_datasets()
        print(datasets)

asyncio.run(main())
```

### VLM inference on a single asset

```python
from eyepop.data.data_types import InferRequest, TranscodeMode

async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
    job = await endpoint.infer_asset(
        asset_uuid='your-asset-uuid',
        infer_request=InferRequest(text_prompt='Describe this image.'),
        transcode_mode=TranscodeMode.image_cover_1024,
    )
    while result := await job.predict():
        print(result)
```

### Batch dataset evaluation

```python
from eyepop.data.data_types import EvaluateRequest, InferRequest

request = EvaluateRequest(
    dataset_uuid='your-dataset-uuid',
    infer=InferRequest(text_prompt='How many people are in this image?'),
)

async with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=4) as endpoint:
    job = await endpoint.evaluate_dataset(evaluate_request=request)
    response = await job.response
    print(response.model_dump_json(indent=2))
```

See [`examples/infer_demo.py`](examples/infer_demo.py) for the complete flow.

## Advanced

### Custom worker image

```python
EyePopSdk.sync_worker(
    pipeline_image='my-registry/my-worker:latest',
    pipeline_version='1.0.0',
)
```

Or set `EYEPOP_PIPELINE_IMAGE` and `EYEPOP_PIPELINE_VERSION`.

### Local mode

Connect to a local worker at `127.0.0.1:8080`:

```python
EyePopSdk.sync_worker(is_local_mode=True)
```

Or set `EYEPOP_LOCAL_MODE=true`.
