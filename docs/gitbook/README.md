---
description: Call EyePop from Python
icon: python
---

# Python SDK

The EyePop Python SDK calls the inference and data APIs from Python. Install it, set an API key, and run a Pop against an image, video, or stream.

```shell
pip install eyepop
```

Requires Python 3.12 or newer.

### Your first prediction

```python
from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

pop = Pop(components=[
    InferenceComponent(ability="eyepop.person:latest")
])

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload("photo.jpg").predict()
    print(result)
```

`upload()` queues the file and `predict()` blocks until the result is ready. A single image yields one prediction; a video or multi-frame container yields one per frame, so call `predict()` in a loop until it returns `None`.

Pass the Pop when you open the session so EyePop can schedule the right compute before any media is processed.

To run on your own hardware instead of EyePop's cloud, point the SDK at an [on-premise instance](https://docs.eyepop.ai/developer-documentation/cli/on-premise) with **local mode**, `is_local_mode=True`. See [Local mode](configuration.md#local-mode).

### Next steps

* [Configuration](configuration.md) — credentials and environment variables
* [Running Inference](inference.md) — files, streams, URLs, video, and image groups
* [Composable Pops](composable-pops.md) — chain models into a pipeline
* [Data Endpoint](data-endpoint.md) — datasets, VLM inference, and evaluation
