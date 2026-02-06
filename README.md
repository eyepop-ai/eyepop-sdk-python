# EyePop.ai Python SDK
The EyePop.ai Python SDK provides convenient access to EyePop.ai's inference and data APIs from Python.

## Requirements
* Python 3.12+

## Install
```shell
pip install eyepop
```

## Configuration

The SDK reads the following environment variables by default:

| Variable | Required | Description |
|---|---|---|
| `EYEPOP_API_KEY` | Yes | API key from your [EyePop dashboard](https://dashboard.eyepop.ai) account. |
| `EYEPOP_POP_ID` | No | Pop Id from your dashboard. Defaults to `"transient"` if not set. |
| `EYEPOP_ACCOUNT_ID` | For Data API | Account UUID, required when using `EyePopSdk.dataEndpoint()`. |

We recommend using [python-dotenv](https://pypi.org/project/python-dotenv/) to load credentials from a `.env` file so they are not stored in source control.

```python
from eyepop import EyePopSdk

endpoint = EyePopSdk.workerEndpoint()
endpoint.connect()
# do work ....
endpoint.disconnect()
```

Or pass credentials explicitly:

```python
endpoint = EyePopSdk.workerEndpoint(
    pop_id='my-pop-id',
    api_key='my-api-key',
)
```

## Usage Examples

### Uploading and processing a single image

```python
from eyepop import EyePopSdk


def upload_photo(file_path: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        result = endpoint.upload(file_path).predict()
        print(result)


upload_photo('examples/example.jpg')
```
1. `EyePopSdk.workerEndpoint()` returns an endpoint object that authenticates and connects to a worker.
2. Using `with ... endpoint:` automatically manages the connection lifecycle. Alternatively, call
`endpoint.connect()` and `endpoint.disconnect()` manually.
3. `endpoint.upload(...)` uploads a local file to the worker. The image is queued and processed when the worker becomes available.
4. `predict()` blocks until the next prediction result and returns it as a dict. For a single image there is one result; for videos or image containers (e.g. GIF), call `predict()` in a loop until it returns `None`.

To upload a binary stream, use `upload_stream()` with the file-like object and MIME type:

```python
from eyepop import EyePopSdk


def upload_photo_from_stream(file_path: str, mime_type: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        with open(file_path, 'rb') as file:
            result = endpoint.upload_stream(file, mime_type).predict()
            print(result)


upload_photo_from_stream('examples/example.jpg', 'image/jpeg')
```

### Visualizing Results
The SDK includes helper classes to visualize bounding boxes with `matplotlib.pyplot`.

```python
from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk

with EyePopSdk.workerEndpoint() as endpoint:
    result = endpoint.upload('examples/example.jpg').predict()
with Image.open('examples/example.jpg') as image:
    plt.imshow(image)
plot = EyePopSdk.plot(plt.gca())
plot.prediction(result)
plt.show()
```
Depending on the environment, you might need an interactive backend (e.g. `pip install pyqt5`).
See [visualize_with_webui2.py](examples/visualize_with_webui2.py) for a more comprehensive visualization example.

### Uploading and processing batches of images
Instead of waiting for each `predict()` before submitting the next job, queue all jobs first and collect results later. This avoids sequential accumulation of HTTP roundtrip time.

```python
from eyepop import EyePopSdk


def upload_photos(file_paths: list[str]):
    with EyePopSdk.workerEndpoint() as endpoint:
        jobs = []
        for file_path in file_paths:
            jobs.append(endpoint.upload(file_path))
        for job in jobs:
            print(job.predict())


upload_photos(['examples/example.jpg'] * 100)
```
### Asynchronous uploading and processing
For large batches, use the `async` variant with `on_ready` callbacks to process results as they arrive:

```python
import asyncio
from eyepop import EyePopSdk
from eyepop import Job


async def async_upload_photos(file_paths: list[str]):
    async def on_ready(job: Job):
        print(await job.predict())

    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
        for file_path in file_paths:
            await endpoint.upload(file_path, on_ready=on_ready)


asyncio.run(async_upload_photos(['examples/example.jpg'] * 100))
```
### Loading images from URLs
Submit a publicly accessible URL for processing instead of uploading a file. Supported protocols:
* HTTP(s) URLs with response Content-Type `image/*` or `video/*`
* RTSP (live streaming)
* RTMP (live streaming)

```python
from eyepop import EyePopSdk


def load_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        result = endpoint.load_from(url).predict()
        print(result)


load_from_url('https://farm2.staticflickr.com/1080/1301049949_532835a8b5_z.jpg')
```
### Processing Videos
Process all frames of a video from a URL (or uploaded file). Call `predict()` in a loop:

```python
from eyepop import EyePopSdk


def load_video_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)


load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```
### Canceling Jobs
Cancel any queued or in-progress job. For example, stop after 10 seconds of video:

```python
from eyepop import EyePopSdk


def load_video_from_url(url: str):
    with EyePopSdk.workerEndpoint() as endpoint:
        job = endpoint.load_from(url)
        while result := job.predict():
            print(result)
            if result['seconds'] >= 10.0:
                job.cancel()


load_video_from_url('https://demo-eyepop-videos.s3.amazonaws.com/test1_vlog.mp4')
```

## Composable Pops

Composable Pops let you build multi-stage inference pipelines by chaining models and processing components together. Use `endpoint.set_pop(pop)` to configure a worker with a custom pipeline at runtime.

### Components

| Component | Purpose |
|---|---|
| `InferenceComponent` | Run an ML model (object detection, classification, segmentation, etc.) |
| `TrackingComponent` | Track detected objects across video frames |
| `ContourFinderComponent` | Extract contours/polygons from segmentation masks |
| `ComponentFinderComponent` | Extract connected components from masks |
| `ForwardComponent` | Route outputs between pipeline stages |

### Forwarding Operators

Components can forward their outputs to sub-components using:

- **`CropForward`** — Crop each detected region and pass it to sub-components for further processing.
- **`FullForward`** — Pass the full image to sub-components (e.g. for segmentation after detection).

Both support `includeClasses` to filter which detections get forwarded.

### Example: Person Detection

```python
import asyncio
from eyepop import EyePopSdk
from eyepop.worker.worker_types import Pop, InferenceComponent

pop = Pop(components=[
    InferenceComponent(
        ability='eyepop.person:latest',
        categoryName="person"
    )
])

async def main():
    async with EyePopSdk.workerEndpoint(is_async=True) as endpoint:
        await endpoint.set_pop(pop)
        job = await endpoint.upload('examples/example.jpg')
        while result := await job.predict():
            print(result)

asyncio.run(main())
```

### Example: Vehicle Detection with License Plate OCR

Chain models together: detect vehicles, crop each one, detect license plates within the crop, then read the text:

```python
from eyepop.worker.worker_types import (
    Pop, InferenceComponent, TrackingComponent,
    CropForward, MotionModel,
)

pop = Pop(components=[
    InferenceComponent(
        ability='eyepop.vehicle:latest',
        categoryName="vehicles",
        confidenceThreshold=0.8,
        forward=CropForward(
            targets=[
                TrackingComponent(
                    maxAgeSeconds=5.0,
                    motionModel=MotionModel.CONSTANT_VELOCITY,
                    agnostic=True,
                ),
                InferenceComponent(
                    ability='eyepop.vehicle.licence-plate:latest',
                    topK=1,
                    forward=CropForward(
                        targets=[InferenceComponent(
                            ability='eyepop.text.recognize.landscape:latest',
                            categoryName="licence-plate"
                        )]
                    )
                )
            ]
        )
    )
])
```

### Example: Object Localization with VLM Prompts

Use VLM abilities with custom prompts for open-vocabulary detection:

```python
from eyepop.worker.worker_types import Pop, InferenceComponent, CropForward

pop = Pop(components=[
    InferenceComponent(
        id=1,
        ability='eyepop.localize-objects:latest',
        params={"prompts": [{"prompt": "person"}]},
        forward=CropForward(
            targets=[InferenceComponent(
                ability='eyepop.image-contents:latest',
                params={"prompts": [{"prompt": "hair color?"}]}
            )]
        )
    )
])
```

See [pop_demo.py](examples/pop_demo.py) for the full set of composable pop examples including face mesh, hand tracking, body pose, text detection, SAM segmentation, and more.

## Data Endpoint

The Data Endpoint provides access to dataset management, VLM inference, and evaluation workflows.

```python
import asyncio
from eyepop import EyePopSdk

async def main():
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        datasets = await endpoint.list_datasets()
        print(datasets)

asyncio.run(main())
```

### VLM Inference on Assets

Run a VLM model against a single asset:

```python
from eyepop.data.data_types import InferRequest, TranscodeMode

infer_request = InferRequest(
    worker_release='qwen3-instruct',
    text_prompt='Describe what you see in this image.',
)

async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
    job = await endpoint.infer_asset(
        asset_uuid='your-asset-uuid',
        infer_request=infer_request,
        transcode_mode=TranscodeMode.image_cover_1024,
    )
    while result := await job.predict():
        print(result)
```

### Batch Dataset Evaluation

Evaluate a VLM model across an entire dataset:

```python
from eyepop.data.data_types import EvaluateRequest, InferRequest

evaluate_request = EvaluateRequest(
    dataset_uuid='your-dataset-uuid',
    infer=InferRequest(
        worker_release='qwen3-instruct',
        text_prompt='How many people are in this image?',
    ),
)

async with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=4) as endpoint:
    job = await endpoint.evaluate_dataset(evaluate_request=evaluate_request)
    response = await job.response
    print(response.model_dump_json(indent=2))
```

See [infer_demo.py](examples/infer_demo.py) for a complete VLM inference and evaluation example.

## Other Options

### Auto start workers
By default, connecting to a worker endpoint will start a worker if none is running. Disable with:
```python
EyePopSdk.workerEndpoint(auto_start=False)
```

### Stop pending jobs
By default, connecting cancels all currently running or queued jobs on the worker (the caller _takes full control_). Disable with:
```python
EyePopSdk.workerEndpoint(stop_jobs=False)
```

### Local mode
For local development against a worker running on `127.0.0.1:8080`:
```python
EyePopSdk.workerEndpoint(is_local_mode=True)
```
Or set the environment variable `EYEPOP_LOCAL_MODE=true`.

## Release Process

The SDK uses [setuptools-scm](https://github.com/pypa/setuptools-scm) for automatic versioning from git tags. To release:

1. Merge your PR to `main`.
2. Go to **Releases** > [Draft a new release](https://github.com/eyepop-ai/eyepop-sdk-python/releases/new).
3. **Create a new tag** with the version number (e.g. `3.10.0`). Use semver: patch for fixes, minor for features, major for breaking changes.
4. Fill in the release title and click **Generate Release Notes**.
5. Click **Publish Release** to trigger the [GitHub Action](https://github.com/eyepop-ai/eyepop-sdk-python/actions) that builds and publishes to PyPI.
