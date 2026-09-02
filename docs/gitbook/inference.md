---
description: Process files, streams, URLs, video, and image groups
icon: play
---

# Running Inference

Every example opens a session with a Pop, submits media, and reads predictions.

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

`load_from()` accepts HTTP, RTSP, and RTMP sources; the server fetches them.

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

A group is a **single** source processed **together** as one inference unit — a multi-image VLM prompt, for example. It returns one prediction for the whole set, unlike batching below, where each image is independent.

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

Image order is preserved end to end. A group holds **up to 16 images**, enforced server-side. The Pop's ability must be multi-image capable; a single-image ability handed a group returns an error.

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

### Next steps

* [Composable Pops](composable-pops.md) — chain models into a pipeline
* [Data Endpoint](data-endpoint.md) — datasets, VLM inference, and evaluation
