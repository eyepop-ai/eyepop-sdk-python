# EyePop.ai Python SDK

Python SDK for EyePop.ai's inference and data APIs.

Full documentation: [docs/gitbook](docs/gitbook/README.md), published at [docs.eyepop.ai](https://docs.eyepop.ai).

```shell
pip install eyepop
```

Requires Python 3.12+.

## Quickstart

```python
from eyepop import EyePopSdk

with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.upload('photo.jpg').predict()
    print(result)
```

Set `EYEPOP_API_KEY` in your environment (get one at [dashboard.eyepop.ai](https://dashboard.eyepop.ai)), or pass `api_key=...` to `sync_worker()`:

```python
endpoint = EyePopSdk.sync_worker(api_key='my-api-key')
```

## Configuration

Credentials are read from environment variables. Set **one** auth method:

| Variable | Description |
|---|---|
| `EYEPOP_API_KEY` | API key from your dashboard. |
| `EYEPOP_ACCESS_TOKEN` | Pre-issued OAuth access token. |
| `EYEPOP_SECRET_KEY` | Secret key for a named pop. |

Optional:

| Variable | Description |
|---|---|
| `EYEPOP_POP_ID` | Named pop ID. Defaults to `transient`. |
| `EYEPOP_ACCOUNT_ID` | Required for some Data API calls. |

## Usage

### Single image

```python
from eyepop import EyePopSdk

with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.upload('photo.jpg').predict()
    print(result)
```

`upload()` queues the file; `predict()` blocks until the result is ready. For videos or multi-frame containers, call `predict()` in a loop until it returns `None`.

### Binary streams

```python
with EyePopSdk.sync_worker() as endpoint:
    with open('photo.jpg', 'rb') as file:
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

### Image groups (multiple images, one result)

Send several images as a **single** source that the pop processes **together** as
one inference unit — for example a multi-image VLM prompt. The group yields
**one** prediction for the whole set, unlike [Batching](#batching) below, where
each image is an independent inference.

```python
with EyePopSdk.sync_worker() as endpoint:
    # local files
    result = endpoint.upload_group(['a.jpg', 'b.jpg', 'c.jpg']).predict()

    # in-memory streams (optional parallel content types)
    with open('a.jpg', 'rb') as a, open('b.jpg', 'rb') as b:
        result = endpoint.upload_stream_group([a, b]).predict()

    # remote URLs (the server fetches each)
    result = endpoint.load_from_group([
        'https://example.com/a.jpg',
        'https://example.com/b.jpg',
    ]).predict()
```

Image order is preserved end-to-end. A group may contain **up to 16 images**
(enforced server-side). The pop's ability must be multi-image-capable; a
single-image ability handed a group returns an error.

### Batching

Queue multiple uploads, then collect results:

```python
file_paths = ['photo1.jpg', 'photo2.jpg']

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

asyncio.run(main(['photo1.jpg', 'photo2.jpg']))
```

### Visualize results

```python
from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk

with EyePopSdk.sync_worker() as endpoint:
    result = endpoint.upload('photo.jpg').predict()

with Image.open('photo.jpg') as image:
    plt.imshow(image)
EyePopSdk.plot(plt.gca()).prediction(result)
plt.show()
```

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
                ability='eyepop.vehicle.license-plate:latest',
                topK=1,
                forward=CropForward(targets=[
                    InferenceComponent(
                        ability='eyepop.text.recognize.landscape:latest',
                        categoryName='license-plate',
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
                ability='my-company.describe-hair-color:latest',
            ),
        ]),
    ),
])
```

## World coordinates

Predictions can carry a 3D position in **metres** alongside their 2D one, back-projected
through a depth map. Two things have to be true: the Pop must name a depth ability, and
the components whose predictions should be translated must opt in.

```python
from eyepop.worker.worker_types import Pop, PopDepthMap, InferenceComponent, SourceDefaults
from eyepop.worker.camera import Camera

pop = Pop(
    components=[
        InferenceComponent(ability='eyepop.person:latest', toWorld=True),
    ],
    depthMap=PopDepthMap(ability='eyepop.depth.anything-3:latest'),
    defaults=SourceDefaults(camera=Camera(hfovDegrees=72.0)),
)
```

Use a **metric** depth ability. A `relative` one is accepted and silently produces no
world coordinates at all: relative depth is scale- *and* shift-invariant, so a cloud
recovered from it would be distorted rather than merely unscaled.

`toWorld` only means something on a component that runs its own inference —
inference and tracking. A contour finder's points do get enriched, but they belong to the
object that fed it, so the request goes on the inference component upstream.

### The whole scene

`PopDepthMap(toWorld=True)` back-projects the depth map itself, so the results carry a
point cloud of the entire scene rather than one per segmented object. It is also what
reveals the map: without it the depth branch the Pop builds stays out of the response.
It stands on its own — a Pop with nothing but a `depthMap` asking for `toWorld` is
complete, no component needs to opt in.

```python
pop = Pop(
    components=[InferenceComponent(ability='eyepop.person:latest')],
    depthMap=PopDepthMap(ability='eyepop.depth.anything-3:latest', toWorld=True),
)
```

The scene cloud arrives as `depth.world`, indexed exactly like `depth.values`: same grid,
same order, so the point for a pixel and the depth it came from share an index. Both are
sent, because a `NaN` point says only that the pixel could not be placed while the value
at that index says why — `+Infinity` for sky, or a reading that is not a distance at all.

### Reading the coordinates

`worldX`, `worldY` and `worldZ` appear on key points, outline points and contour points
(including cutouts). They are **prediction v2 only**. A point the worker could not place —
sky, outside the depth map, no usable map — carries none of the three, so test for `None`
rather than for a sentinel value:

```python
for keypoints in prediction['keyPoints']:
    for point in keypoints['points']:
        if point.get('worldZ') is not None:
            print(point['worldX'], point['worldY'], point['worldZ'])
```

`z` and `worldZ` are unrelated: `z` is model-relative depth in whatever convention the
model uses, `worldZ` is metres. Bounding boxes are not enriched — a box is not a point,
and any single anchor choice would be arbitrary.

An object with a segmentation mask also carries a dense point cloud, one xyz triple per
mask pixel, which `eyepop.PointCloud` decodes:

```python
from eyepop import PointCloud

cloud = PointCloud.from_object(obj)
if cloud is not None:
    print(cloud.array.shape)        # (height, width, 3), float32, NaN where unplaced
    print(cloud.at(0, 0))           # by mask pixel, or None
    print(cloud.at_source(x, y))    # by source coordinate inside the object's box
    print(cloud.placed_points)      # (N, 3), just the points that were placed
    print(cloud.bounds)             # per-axis (min, max) in metres, or None
```

`PointCloud.from_depth(depth, source_width, source_height)` reads the scene cloud the same
way; there `.at(i, j)` indexes the depth map's own grid, and the frame is the box that
`.at_source(x, y)` maps into.

`PointCloud.from_prediction(prediction)` returns every cloud in one prediction — a list,
not a single value like `DepthMap.from_prediction`, because a cloud belongs to one object's
mask. The scene cloud, if there is one, comes last.

To see the world coordinates — not just the clouds, but key points, outlines and contours
too — scatter them into a 3D axes:

```python
import matplotlib.pyplot as plt
from eyepop.visualize import EyePopWorldPlot

axes = plt.figure().add_subplot(projection='3d')
plot = EyePopWorldPlot(axes)
plot.prediction(prediction)
plot.finish()
plt.show()
```

Each carrier is its own colour and legend entry, and the ones with a defined order are
connected: contours, outlines and cutouts by their own point order, closed; key points by
the pose skeleton, matched on class label from the same table the 2D renderer uses. An
unplaced point breaks the path rather than being bridged over, and a key point group whose
category is not a known pose gets no lines at all rather than points joined in arrival
order. A mask point cloud is a grid rather than a path, so it stays unconnected.

The point budget is shared across series, so adding objects thins the scatter rather than
multiplying it — but sparse series are exempt, so a 17-point skeleton survives beside a
40,000-point mask cloud instead of being thinned away to save a fraction of it.

Z is drawn into the scene and Y up the page, rather than in component order, since depth
is what reads as distance from the viewer in either frame. The coordinates themselves are
never altered.

The vertical axis is also flipped, so a camera-frame scene stands the right way up. Y
grows **downwards** in the OpenCV camera frame, so drawn as-is a figure's feet sit above
its head. That is the default because a source supplying no extrinsics — or identity
ones — gets the camera frame. For coordinates already in a Z-up world frame, pass
`EyePopWorldPlot(axes, invert_y=False)`.

### Camera calibration

Without a calibration the worker falls back to an assumed 60° horizontal field of view.
That is a development scaffold, not something to ship: for canonical metric depth the
guess cancels out of X and Y and survives only in Z, so lateral measurements stay exact
while every distance along the optical axis is wrong by however wrong the guess was.

Supply a `camera` per source, or once for every source through the Pop's `defaults`:

```python
from eyepop.worker.camera import Camera, CameraIntrinsics, CameraExtrinsics, Vector3d

camera = Camera(
    intrinsics=CameraIntrinsics(fx=0.9, fy=1.6, cx=0.5, cy=0.5),
    extrinsics=CameraExtrinsics(translation=Vector3d(z=3.0)),
)
endpoint.load_from(url, camera=camera)
```

Exactly one of `intrinsics` and `hfovDegrees` describes the lens. Both is rejected rather
than resolved by precedence, and neither is rejected too, since defaulting a focal length
would be inventing a lens.

**Intrinsics are normalised to the frame, not given in pixels** — `fx`/`fy` as a fraction
of the frame's width and height — so one calibration survives a resolution change.

**Extrinsics are camera to world**: `P_world = R * P_camera + t`, so `t` is where the
camera sits. This is the inverse of what `cv2.solvePnP` returns; a caller holding its
`rvec`/`tvec` must invert both halves, and `tvec` is *not* the camera position. With
extrinsics, coordinates are in the world frame those define — Z up, ground at Z = 0.
Without them, they are in the camera frame, OpenCV convention: X right, Y down, Z forward.

Defaults merge per field, so a source giving its own `roi` but no `camera` keeps its roi
and takes the default camera.

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
