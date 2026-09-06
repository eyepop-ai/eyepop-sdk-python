---
description: Datasets, VLM inference, and batch evaluation
icon: database
---

# Data Endpoint

The Data API manages datasets and runs VLM inference and evaluation over them.

```python
import asyncio
from eyepop import EyePopSdk

async def main():
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        datasets = await endpoint.list_datasets()
        print(datasets)

asyncio.run(main())
```

Some Data API calls need an account: set `EYEPOP_ACCOUNT_ID`. See [Configuration](configuration.md).

{% hint style="warning" %}
`infer_asset` and `evaluate_dataset` are experimental and may change without a major version bump.
{% endhint %}

### VLM inference on one asset

```python
from eyepop.data.data_types import InferRequest, TranscodeMode

async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
    job = await endpoint.infer_asset(
        asset_uuid="your-asset-uuid",
        infer_request=InferRequest(text_prompt="Describe this image."),
        transcode_mode=TranscodeMode.image_cover_1024,
    )
    while result := await job.predict():
        print(result)
```

### Batch dataset evaluation

```python
from eyepop.data.data_types import EvaluateRequest, InferRequest

request = EvaluateRequest(
    dataset_uuid="your-dataset-uuid",
    infer=InferRequest(text_prompt="How many people are in this image?"),
)

async with EyePopSdk.dataEndpoint(is_async=True, job_queue_length=4) as endpoint:
    job = await endpoint.evaluate_dataset(evaluate_request=request)
    response = await job.response
    print(response.model_dump_json(indent=2))
```

### Next steps

* [Composable Pops](composable-pops.md) — chain models into an inference pipeline for the worker endpoint
* [Running Inference](inference.md) — process media directly instead
