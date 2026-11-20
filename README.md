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
from eyepop.worker.worker_types import InferenceComponent, Pop

pop = Pop(components=[InferenceComponent(ability='eyepop.person:latest')])

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload('photo.jpg').predict()
    print(result)
```

Set `EYEPOP_API_KEY` in your environment (get one at [dashboard.eyepop.ai](https://dashboard.eyepop.ai)), or pass `api_key=...` to `sync_worker()`:

```python
endpoint = EyePopSdk.sync_worker(api_key='my-api-key', pop=pop)
```

## Configuration

Credentials are read from environment variables. Set **one** auth method:

| Variable | Description |
|---|---|
| `EYEPOP_API_KEY` | API key from your dashboard. |
| `EYEPOP_ACCESS_TOKEN` | Pre-issued OAuth access token. |
| `EYEPOP_SECRET_KEY` | Secret key for a named pop. Worker session only: see [Pipeline mode](#pipeline-mode). |

Optional:

| Variable | Description |
|---|---|
| `EYEPOP_POP_ID` | Named pop ID. Defaults to `transient`. Worker session only. |
| `EYEPOP_PIPELINE_MODE` | Pipeline mode is on by default. Set to `false` to run on a worker session instead. |
| `EYEPOP_ENV` | Pipeline mode: `staging`, `production` or `local` picks the pipeline API and the compute API that mints the token. `pipeline_api_url` wins over it for the pipeline API. A named compute API (`eyepop_url` or `EYEPOP_URL`) means `EYEPOP_ENV` is not read: that compute API names the pipeline API in its `/v1/configs`, unless `pipeline_api_url` is given. |
| `EYEPOP_ACCOUNT_UUID` | The account to work in. Required for some Data API calls, and names the account a worker's compute session runs for when the API key does not (same as `account_id=`). `EYEPOP_ACCOUNT_ID` is still read when this is unset, with a `DeprecationWarning`. |

## Usage

The snippets below use the `pop` from the Quickstart.

### Single image

```python
from eyepop import EyePopSdk

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload('photo.jpg').predict()
    print(result)
```

`upload()` queues the file; `predict()` blocks until the result is ready. For videos or multi-frame containers, call `predict()` in a loop until it returns `None`.

### Binary streams

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    with open('photo.jpg', 'rb') as file:
        result = endpoint.upload_stream(file, 'image/jpeg').predict()
```

### URLs (HTTP, RTSP, RTMP)

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.load_from('https://example.com/image.jpg').predict()
```

### Videos

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
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
with EyePopSdk.sync_worker(pop=pop) as endpoint:
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

with EyePopSdk.sync_worker(pop=pop) as endpoint:
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

    async with EyePopSdk.async_worker(pop=pop) as endpoint:
        for p in paths:
            await endpoint.upload(p, on_ready=on_ready)

asyncio.run(main(['photo1.jpg', 'photo2.jpg']))
```

### Visualize results

```python
from PIL import Image
import matplotlib.pyplot as plt
from eyepop import EyePopSdk

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload('photo.jpg').predict()

with Image.open('photo.jpg') as image:
    plt.imshow(image)
EyePopSdk.plot(plt.gca()).prediction(result)
plt.show()
```

### Recordings: thumbnails and clips by mediaId

A source with `media_cache_seconds` records to the media cache. Once it has started, its job's
`media_id` names the recording: an opaque string you pass back unchanged. Read a frame, a clip or the
range still held through the endpoint, in either mode:

```python
job = endpoint.load_from("rtsp://camera.example.com/stream1", media_cache_seconds=3600)
for _ in range(300):  # a live source never ends on its own: take its first 300 results
    if (result := job.predict()) is None:
        break
    for person in result.get("objects", []):
        # the frame the prediction was made on, cropped to the object's box, fit to 160 x 160, JPEG
        thumbnail = job.media_frame(result, person, max_width=160, max_height=160)

start_ns, end_ns = endpoint.media_range(job.media_id)
with endpoint.media_clip(job.media_id, max(start_ns, end_ns - 10_000_000_000), end_ns) as clip:  # the last 10 s, MP4
    with open("clip.mp4", "wb") as file:
        for chunk in clip:
            file.write(chunk)
job.cancel()  # stop the live source
```

| Method | Returns |
|---|---|
| `media_range(media_id)` | `(start_ns, end_ns)`: the first and last nanosecond the cache still holds. |
| `media_frame(media_id, timestamp_ns, *, bbox=None, max_width=None, max_height=None, png=False, wait_ms=None)` | The frame at a prediction's `timestamp` as JPEG bytes, PNG with `png=True`. `bbox` is `(x, y, width, height)` in the frame's pixels, sent as the smallest box of whole pixels that holds it and clipped to the frame; `max_width` and `max_height` fit the answer, aspect kept, never enlarged. |
| `media_clip(media_id, start_ns, end_ns, *, mode="exact", wait_ms=None)` | A `MediaClip` that streams an MP4: iterate it for its bytes or `read()` it whole, and close it (`with`) when you stop early. `mode="exact"` re-encodes to the exact bounds (300 s at most by default); `"keyframe"` copies from the keyframe at or before `start_ns`. `start_ns` and `end_ns` are the clip's real bounds; `milli_compute_units` is what the pipeline API priced it at, None when the answer carries none. |
| `job.media_frame(prediction, obj=None, **options)` | `media_frame` with the job's `media_id`, the prediction's `timestamp` and the object's `x`, `y`, `width` and `height` (source pixels, which are the recording's). `ValueError` when the job has no `media_id`. |

The async endpoint has the same methods as coroutines (`await endpoint.media_frame(...)`, `async with
await endpoint.media_clip(...) as clip`, `async for chunk in clip`). A session's endpoint reads from its
instance, which knows only its own session's recordings (`v2.` ids, from the `recording` event that opens
a recording source's results); a pipeline endpoint reads from the pipeline API, which knows every
recording of the token's account, a session's too once its instance is gone. A frame read right after its
prediction waits on the server for the moment to be cached (`wait_ms`, 2000 by default, at most 10000);
an answer `425`, or `503 overloaded` from a pipeline API at capacity, is asked again after its
`Retry-After` for up to `EYEPOP_MAX_RETRY_TIME_SECS` (30 s), each retry bounded by what is left of it.
Any other refusal raises `MediaReadException` with the `status` and the answer's `error`, `reason` (the
code to act on, such as `stream_not_found`, `media_expired` or `clip_too_long`) and `message`. The pipeline
API prices frame and clip reads in milli compute units, in dry run; a session's reads are part of its
session minutes. Needs eyepop-instance with spec 025 on the instance or the pipeline API.

A job also keeps the pipeline's `warning` events on `job.warnings`, in both modes, and logs each at
`WARNING` on `eyepop`; one says a recorded camera sends keyframes too rarely
(`media_cache_gop_too_long`), which `media_cache_reencode=True` avoids.

## Pipeline mode

A worker endpoint runs its jobs on the pipeline API by default, not on a worker session. Pass
`is_pipeline_mode=False`, or set `EYEPOP_PIPELINE_MODE=false`, to use a worker session, which is deprecated and warns with a `DeprecationWarning`; the argument
wins over the environment. Code that names a session (`secret_key`, a `pop_id` other than `"transient"`, `session_uuid`,
`session_name`, `pipeline_image`, `pipeline_version`, `dataset_uuid`, or `EYEPOP_SESSION_UUID` in the
environment) raises `ValueError` in pipeline mode and needs the opt-out. Pass `pop=` (or call `set_pop()` before the first job): a job with no pop
raises `ValueError`, since a pipeline with no pop runs nothing.

The code that uses the endpoint stays the same:

```python
from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

pop = Pop(components=[InferenceComponent(ability='eyepop.person:latest')])
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    print(endpoint.upload('photo.jpg').predict())
```

The endpoint opens a socket to the pipeline API at `connect()` and sets its pop there. `eyepop_url`
(or `EYEPOP_URL`) is the compute API, which mints the token. `pipeline_api_url`, or `EYEPOP_ENV`
(`staging`, `production` or `local`, which picks both URLs), names the pipeline API; unset, the compute
API names it in its `/v1/configs`. With `is_local_mode`, `eyepop_url` is the pipeline API itself
(default `http://127.0.0.1:9002`). Jobs go
out in the order they are submitted, each queued behind the ones before it on its socket, up to 256
at a time: `job_queue_length` defaults to 256 and may not exceed it.

Each socket runs its own pipeline, which plays its sources one after another. While every open
socket has a source, the next job opens another, up to 4 (`EYEPOP_PIPELINE_SOCKETS`), each with the
pop set; a job goes on the socket with the fewest open sources. A batch submitted at once therefore
runs on 4 pipelines, and jobs run one after another stay on one socket. On a T4, 4 sockets of
704-pixel images ran `common-objects.large` at about 89 images per second, against 30 on one. When
the pipeline API refuses another socket, the job goes on an open one, and the endpoint asks again
only 10 s later. A live source never ends, so nothing queued behind it would play: it goes on an
idle socket or a new one, which the 4 do not count, and no other job is sent on a socket that has
one. `set_pop` sets the pop on every open socket.

**Waiting for capacity.** A pipeline API with no capacity refuses a new socket with HTTP 503 (or
429). A new GPU node takes minutes to arrive, so the SDK waits and asks again, for up to
`EYEPOP_PIPELINE_WAIT_TIMEOUT` seconds (default 600, `pipeline_wait_timeout` in
`eyepop.settings.settings`; `0` does not wait). Each wait is the `Retry-After` the answer names (at
least 1 s) with 0 to 50 % more, so that sockets do not retry in step; a `Retry-After` that repeats
grows by half each time, up to 10 s (or itself, when it asks for more). Without `Retry-After` the
waits are 1, 2, 4, 8 and then 10 s, each between half and all of it. The first wait is logged at
`WARNING` on `eyepop.requests` (the status, its `Retry-After` and the time left), each retry at
`DEBUG`. A wait that would pass the limit is not started and no try is sent before `Retry-After`
passed: the SDK raises `PipelineOverloadedException` (a `PipelineCommandException`; tell it from a
refused source with `isinstance`) with the refusal as `__cause__`. It is not an `aiohttp.ClientError`,
so a caller that caught `aiohttp.ClientResponseError` for a 503 or 429 must catch it instead, also
with `0`. Jobs queued behind a wait that gave up fail with it at once, instead of starting a wait
each, so a queue of any length fails at once; a job submitted after the give-up tries and waits
for itself, since capacity may be back. A socket that opens, `connect()` and
`disconnect()` clear the record. The error carries the give-up's own `waited_secs`, 0 when it did not
wait. Whatever `Retry-After` says, a wait that goes on settles at 10 to 15 s between tries, so a
client may notice a new replica up to 15 s after it came up. `disconnect()`, and the end of
`async with`, end a wait at once with `PipelineSocketClosedException`, and cancelling a job ends its own wait and lets go of the send lock. A wait holds the endpoint's
send lock, so jobs submitted after it queue behind it. Any other refusal, such as a 401 (the API key is exchanged once more) or a 404,
raises at once. This holds for `connect()` and for any job or reconnect that opens a socket, except
the extra sockets of a batch, which the job does not wait for (see above). A script or a test suite
that should fail fast on a full pipeline API sets `EYEPOP_PIPELINE_WAIT_TIMEOUT=0` or a few seconds.

| | Pipeline mode |
|---|---|
| `upload`, `upload_group` | Files up to 8 MiB go in one message (`EYEPOP_COMBINED_UPLOAD_MAX_BYTES`), larger ones in chunks. |
| `upload_stream`, `upload_stream_group`, `upload(is_live=True)` | Sent in chunks as the pipeline reads them. |
| `load_from`, `load_from_group`, `load_asset` | Supported. An `rtsp://` or `rtmp://` url is live. |
| `params`, `roi`, `camera`, `fps` | Supported. |
| `motion_detect` | Sent with `setSource` (`motionDetect`, `motionSensitivity`, `motionThreshold`, `motionGap`, `motionGridX`, `motionGridY`). Needs a pipeline API on eyepop-worker 4.15.0 or later. A group takes none: `ValueError` at submission. |
| `video_mode` | Sent with a single upload (`upload`, `upload_stream`); unset, a video upload that is not live is buffered. Needs eyepop-worker 4.15.0 or later. Not sent for any other source; a warning names it once per endpoint. |
| `media_cache_seconds`, `media_cache_reencode` | Sent with `setSource`; a source that records gets its `media_id` from its `start`. A group takes neither: `ValueError` at submission. |
| `rtsp_force_non_compliant_url` | Not sent; a warning names it once per endpoint. |
| `set_pop`, `get_pop` | Supported. A pop the pipeline refuses raises `PopConfigurationException`, and the last pop it took stays. |
| `session()` | Returns `{"pipeline_api_url": ...}`. |
| `pipeline_get/post/patch/delete`, `dev_mode_base_url`, `dev_mode_pipeline_base_url` | Raise `NotImplementedError`: there is no worker session. |
| `api_key`, `access_token` | An API key is exchanged at the compute API and never sent to the pipeline API; an access token is sent as given. |
| `secret_key`, a `pop_id` other than `"transient"`, `session_uuid`, `session_name`, `pipeline_image`, `pipeline_version` | Raise `ValueError` at construction. |
| `dataset_uuid` | Raises `ValueError` at construction until the pipeline API writes to datasets (AWSU-316). |
| Prediction version | The pipeline API's own (V3); none is sent. |

**Errors.** A source that ends with an error raises `PipelineSourceException` (a `ValueError`) from
`predict()`. A command the pipeline API refuses raises `PipelineCommandException`, except
`queue_full` on a source: the socket is full (256 open sources, or 32 MiB of images sent in one
message), so the source goes again once another of the endpoint's sources on that socket ended. It
raises only when none of them is open. When the socket closes under a job, the job raises
`PipelineSocketClosedException` with the close's code and reason and is not sent again; jobs not
yet sent go out on a new socket. After the pipeline stops itself when idle, the next job opens a
new socket and sets the pop again; after `credential_expired`, `overloaded` or `shutting_down`, a
new socket is opened at once. A live source refused `overloaded` is sent once more on the new
socket, and when that one is refused too, raises `PipelineOverloadedException` (a
`PipelineCommandException`, as is the one a socket that stayed refused raises above); the SDK does
not send it a third time. Cancelling a job stops its source on the pipeline with `stopSource`, queued
or playing, so a cancelled live source frees its place; a pipeline API from before `stopSource` lets
the source run until it ends or the endpoint disconnects.

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

Predictions can carry a 3D position in **meters** alongside their 2D one, back-projected
through a depth map. Two things have to be true: the Pop must name a depth ability, and
the components whose predictions should be translated must opt in.

```python
from eyepop.worker.worker_types import Pop, PopDepthMap, InferenceComponent, SourceDefaults
from eyepop.worker.camera import Camera

pop = Pop(
    components=[
        InferenceComponent(ability='eyepop.person:latest', toWorld=True),
    ],
    depthMap=PopDepthMap(ability='eyepop.depth.metric.small:latest'),
    defaults=SourceDefaults(camera=Camera(hfovDegrees=72.0)),
)
```

Use a **metric** depth ability. A `relative` one is accepted and silently produces no
world coordinates at all: relative depth is scale- *and* shift-invariant, so a cloud
recovered from it would be distorted rather than merely unscaled.

The four metric depth abilities are `eyepop.depth.metric.small`,
`eyepop.depth.metric.small-landscape`, `eyepop.depth.metric.large` and
`eyepop.depth.metric.large-landscape`. A map keeps the source's aspect ratio; the
ability sets the box it fits inside - 280x280 and 518x518 for the plain variants,
504x280 and 924x518 for the `-landscape` ones. Landscape media is therefore much
denser from a `-landscape` variant (1280x720 gives 518x291 against 921x518), while
portrait and square media get the same grid either way. Smaller maps mean faster
responses and higher throughput.

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
    depthMap=PopDepthMap(ability='eyepop.depth.metric.small:latest', toWorld=True),
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
model uses, `worldZ` is meters. Bounding boxes are not enriched — a box is not a point,
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
    print(cloud.bounds)             # per-axis (min, max) in meters, or None
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
    async with EyePopSdk.async_data() as endpoint:
        datasets = await endpoint.list_datasets()
        print(datasets)

asyncio.run(main())
```

### VLM inference on a single asset

```python
import asyncio

from eyepop.data.data_types import InferRequest, TranscodeMode

async def main():
    async with EyePopSdk.async_data() as endpoint:
        job = await endpoint.infer_asset(
            asset_uuid='your-asset-uuid',
            infer_request=InferRequest(text_prompt='Describe this image.'),
            transcode_mode=TranscodeMode.image_cover_1024,
        )
        while result := await job.predict():
            print(result)

asyncio.run(main())
```

### Batch dataset evaluation

```python
import asyncio

from eyepop.data.data_types import EvaluateRequest, InferRequest

request = EvaluateRequest(
    dataset_uuid='your-dataset-uuid',
    infer=InferRequest(text_prompt='How many people are in this image?'),
)

async def main():
    async with EyePopSdk.async_data(job_queue_length=4) as endpoint:
        job = await endpoint.evaluate_dataset(evaluate_request=request)
        response = await job.response
        print(response.model_dump_json(indent=2))

asyncio.run(main())
```
