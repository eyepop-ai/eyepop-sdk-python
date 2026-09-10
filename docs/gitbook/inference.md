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

### Camera calibration

Every upload and load method takes a `camera`, which is what lets a [depth map](../../platform/depth-and-world-coordinates/depth-maps.md) become positions in metres:

```python
from eyepop.worker.camera import Camera

job = endpoint.load_from("rtsp://camera.example.com/stream1", camera=Camera(hfovDegrees=72.0))
```

Set it once for every source with `Pop.defaults` instead — see [Composable Pops](composable-pops.md#world-coordinates).

`EyePopPlot.depth(result)` overlays a frame's depth map as a heatmap, and `EyePopWorldPlot` scatters world coordinates into a 3D axes.

### Next steps

* [Sources and Options](../../platform/sources-and-options/README.md) — every source the platform accepts, and the options that shape processing
* [Composable Pops](composable-pops.md) — chain models into a pipeline
* [Depth and World Coordinates](../../platform/depth-and-world-coordinates/README.md) — depth maps, calibration, and metres
* [Data Endpoint](data-endpoint.md) — datasets, VLM inference, and evaluation
