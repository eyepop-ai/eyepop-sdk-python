---
description: Chain models into a multi-stage inference pipeline
icon: diagram-project
---

# Composable Pops

A Pop chains models into a pipeline: detect, crop to each detection, and run another model on the crop. Pass it when you open the session.

### Components

| Component | Purpose |
| --- | --- |
| `InferenceComponent` | Run a model. Supports chunked video via `videoChunkLengthSeconds` and `videoChunkOverlap`. |
| `TrackingComponent` | Track detected objects across frames. |
| `ContourFinderComponent` | Extract contours from segmentation masks. |
| `ComponentFinderComponent` | Extract connected components from masks. |
| `ForwardComponent` | Route outputs between stages. |

### Forwarding

* `CropForward` — pass each detection crop to sub-components.
* `FullForward` — pass the full image to sub-components.

Both accept `includeClasses` to filter which detections are forwarded.

### Vehicle to license plate to OCR

```python
from eyepop.worker.worker_types import (
    Pop, InferenceComponent, TrackingComponent, CropForward, MotionModel,
)

pop = Pop(components=[
    InferenceComponent(
        ability="eyepop.vehicle:latest",
        categoryName="vehicles",
        confidenceThreshold=0.8,
        forward=CropForward(targets=[
            TrackingComponent(
                maxAgeSeconds=5.0,
                motionModel=MotionModel.CONSTANT_VELOCITY,
                agnostic=True,
            ),
            InferenceComponent(
                ability="eyepop.vehicle.license-plate:latest",
                topK=1,
                forward=CropForward(targets=[
                    InferenceComponent(
                        ability="eyepop.text.recognize.landscape:latest",
                        categoryName="license-plate",
                    ),
                ]),
            ),
        ]),
    ),
])
```

### VLM open-vocabulary detection

```python
from eyepop.worker.worker_types import Pop, InferenceComponent, CropForward

pop = Pop(components=[
    InferenceComponent(
        ability="eyepop.localize-objects:latest",
        params={"prompts": [{"prompt": "person"}]},
        forward=CropForward(targets=[
            InferenceComponent(
                ability="eyepop.image-contents:latest",
                params={"prompts": [{"prompt": "hair color?"}]},
            ),
        ]),
    ),
])
```

### Next steps

* [Running Inference](inference.md) — submit media to the Pop you just built
* [Data Endpoint](data-endpoint.md) — datasets, VLM inference, and evaluation
