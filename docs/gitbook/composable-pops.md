---
description: Build a Pop with the Python types
icon: diagram-project
---

# Composable Pops

A Pop chains abilities into a pipeline: detect, crop to each detection, and run another ability on the crop. Pass it when you open the session.

This page is the Python construction API. Every component type and its attributes are covered once in [Components](../../platform/pop-components.md), how they chain in [Forwarding](../../platform/pop-forwarding.md), and worked pipelines in [Examples](../../platform/pop-examples.md).

### The types

Import them from `eyepop.worker.worker_types`.

| Type | Purpose |
| --- | --- |
| `Pop` | The pipeline itself: `components`, and optionally `postTransform`, `defaults` and `depthMap`. |
| `InferenceComponent` | Run an ability. |
| `TrackingComponent` | Track detected objects across video frames. |
| `ContourFinderComponent` | Turn segmentation masks into contours. `contourType` is optional and defaults to `polygon`. |
| `ComponentFinderComponent` | Split segmentation masks into sub-objects. |
| `ForwardComponent` | Route output onward without analyzing it. |

`InferenceType`, `MotionModel`, and `ContourType` are enums for the corresponding fields.

The components are Pydantic models, so a Pop is validated as you build it rather than when the worker rejects it.

### Forwarding

`CropForward` and `FullForward` are helpers that build the forward operator for you:

```python
CropForward(targets, maxItems=None, boxPadding=None,
            orientationTargetAngle=None, includeClasses=None,
            is_full_fallback=False)

FullForward(targets, includeClasses=None)
```

`is_full_fallback=True` selects `crop_with_full_fallback`, which crops when the parent detected something and falls back to the whole frame when it did not.

### Building a Pop

```python
from eyepop import EyePopSdk
from eyepop.worker.worker_types import (
    Pop, InferenceComponent, TrackingComponent, CropForward, MotionModel,
)

pop = Pop(components=[
    InferenceComponent(
        ability="eyepop.vehicle:latest",
        categoryName="vehicles",
        confidenceThreshold=0.8,
        forward=CropForward(
            includeClasses=["car", "truck"],
            targets=[
                TrackingComponent(
                    maxAgeSeconds=5.0,
                    motionModel=MotionModel.CONSTANT_VELOCITY,
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
            ],
        ),
    ),
])

with EyePopSdk.sync_worker(pop=pop) as endpoint:
    result = endpoint.upload("street.jpg").predict()
```

### World coordinates

`PopDepthMap` names the depth ability, and `toWorld` on a component asks for its point-based predictions in meters. `SourceDefaults` carries a `Camera` for every source the Pop processes.

```python
from eyepop.worker.camera import Camera
from eyepop.worker.worker_types import (
    InferenceComponent, Pop, PopDepthMap, SourceDefaults,
)

pop = Pop(
    components=[InferenceComponent(ability="eyepop.person:latest", toWorld=True)],
    depthMap=PopDepthMap(ability="eyepop.depth.metric.small:latest"),
    defaults=SourceDefaults(camera=Camera(hfovDegrees=72.0)),
)
```

`PopDepthMap` and `Camera` validate as you build them, so a Pop that cannot mean what it says fails here rather than as a `400` from the worker. Decode the results with `eyepop.DepthMap` and `eyepop.PointCloud`, and plot them with `eyepop.visualize.EyePopWorldPlot`.

See [Depth and World Coordinates](../../platform/depth-and-world-coordinates/README.md) for the whole feature.

### Prompting an ability

Abilities backed by a vision-language model take their instruction through `params`:

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

{% hint style="info" %}
`multiClass` is accepted by the platform but is not yet exposed on `InferenceComponent`. Everything else in [Components](../../platform/pop-components.md) is available from Python.
{% endhint %}

### Next steps

* [Components](../../platform/pop-components.md) — every component type and attribute
* [Forwarding](../../platform/pop-forwarding.md) — how components chain
* [Examples](../../platform/pop-examples.md) — worked pipelines end to end
* [Running Inference](inference.md) — submit media to the Pop you just built
* [Depth and World Coordinates](../../platform/depth-and-world-coordinates/README.md) — predictions positioned in meters
* [Data Endpoint](data-endpoint.md) — datasets, VLM inference, and evaluation
