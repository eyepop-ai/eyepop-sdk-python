---
description: Process files, streams, URLs, video, and image groups
icon: play
---

# Running Inference

Every example opens a session with a Pop, submits media, and reads predictions.

This page is the Python call shapes. What a source is, every kind the platform accepts, and the options that shape how one is processed are covered once in [Sources and Options](../../platform/sources-and-options/README.md).

### A single image

```python
from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

pop = Pop(components=[InferenceComponent(ability="eyepop.person:latest")])

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload("photo.jpg").predict()
    print(result)
```

### Binary streams

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    with open("photo.jpg", "rb") as file:
        result = endpoint.upload_stream(file, "image/jpeg").predict()
```

### URLs

`load_from()` hands the platform a URL and lets it fetch, so nothing uploads from your application. [Source Types](../../platform/sources-and-options/sources.md) lists every scheme it accepts.

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.load_from("https://example.com/image.jpg").predict()
```

### Video

A video yields one prediction per frame, so read until `predict()` returns `None`.

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    job = endpoint.load_from("https://example.com/video.mp4")
    while result := job.predict():
        print(result)
```

Cancel a job mid-stream with `job.cancel()`.

### Image groups

A [group](../../platform/sources-and-options/sources.md#image-groups) is one source processed together as a single inference unit, returning one prediction for the whole set — unlike batching below, where each image is independent.

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    # local files
    result = endpoint.upload_group(["a.jpg", "b.jpg", "c.jpg"]).predict()

    # in-memory streams
    with open("a.jpg", "rb") as a, open("b.jpg", "rb") as b:
        result = endpoint.upload_stream_group([a, b]).predict()

    # remote URLs
    result = endpoint.load_from_group([
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
    ]).predict()
```

[Image groups](../../platform/sources-and-options/sources.md#image-groups) covers the size limit, the ordering guarantee, and which abilities accept a group.

### Batching

Queue several uploads, then collect the results. Each image is an independent inference.

```python
with EyePopSdk.sync_worker(pop=pop) as endpoint:
    jobs = [endpoint.upload(p) for p in ["photo1.jpg", "photo2.jpg"]]
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

asyncio.run(main(["photo1.jpg", "photo2.jpg"]))
```

### Visualizing results

```python
from PIL import Image
import matplotlib.pyplot as plt

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload("photo.jpg").predict()

with Image.open("photo.jpg") as image:
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

### Camera calibration

Every upload and load method takes a `camera`, which is what lets a [depth map](../../platform/depth-and-world-coordinates/depth-maps.md) become positions in meters:

```python
from eyepop.worker.camera import Camera

job = endpoint.load_from("rtsp://camera.example.com/stream1", camera=Camera(hfovDegrees=72.0))
```

Set it once for every source with `Pop.defaults` instead — see [Composable Pops](composable-pops.md#world-coordinates).

`EyePopPlot.depth(result)` overlays a frame's depth map as a heatmap, and `EyePopWorldPlot` scatters world coordinates into a 3D axes.

### Next steps

* [Sources and Options](../../platform/sources-and-options/README.md) — every source the platform accepts, and the options that shape processing
* [Composable Pops](composable-pops.md) — chain models into a pipeline
* [Depth and World Coordinates](../../platform/depth-and-world-coordinates/README.md) — depth maps, calibration, and meters
* [Data Endpoint](data-endpoint.md) — datasets, VLM inference, and evaluation
