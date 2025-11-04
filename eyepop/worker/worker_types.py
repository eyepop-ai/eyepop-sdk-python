import enum
from typing import List, Literal, Annotated, Union, Any

from pydantic import BaseModel, Field, ConfigDict

class PredictionVersion(enum.IntEnum):
    V1 = 1
    V2 = 2


DEFAULT_PREDICTION_VERSION = PredictionVersion.V2


class VideoMode(enum.StrEnum):
    STREAM = "stream"
    BUFFER = "buffer"


class PopComponentType(enum.StrEnum):
    BASE = "<invalid>"
    FORWARD = "forward"
    INFERENCE = "inference"
    # backward compatibility for persisted Pops < 3.0.0
    TRACING = "tracing"
    # since 3.0.0, replaced 'tracing'
    TRACKING = "tracking"
    CONTOUR_FINDER = "contour_finder"
    COMPONENT_FINDER = "component_finder"


class ForwardOperatorType(enum.StrEnum):
    FULL = "full"
    CROP = "crop"
    CROP_WITH_FULL_FALLBACK = "crop_with_full_fallback"


class PopCrop(BaseModel):
    maxItems: int | None = None
    boxPadding: float | None = None
    orientationTargetAngle: float | None = None
    model_config = ConfigDict(extra='forbid')


class PopForwardOperator(BaseModel):
    type: ForwardOperatorType
    includeClasses: list[str] | None = None
    crop: PopCrop | None = None
    model_config = ConfigDict(extra='forbid')

class PopForward(BaseModel):
    operator: PopForwardOperator | None = None
    targets: List["DynamicComponent"] | None = None
    model_config = ConfigDict(extra='forbid')

class BaseComponent(BaseModel):
    type: Literal[PopComponentType.BASE] = PopComponentType.BASE
    id: int | None = None
    forward: PopForward | None = None
    model_config = ConfigDict(extra='forbid')


class ForwardComponent(BaseComponent):
    type: Literal[PopComponentType.FORWARD] = PopComponentType.FORWARD
    model_config = ConfigDict(extra='forbid')


class InferenceType(enum.StrEnum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    KEY_POINTS = "key_points"
    OCR = "ocr"
    MESH = "mesh"
    FEATURE_VECTOR = "feature_vector"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    SEGMENTATION = "segmentation"
    RAW = "raw"


class InferenceComponent(BaseComponent):
    type: Literal[PopComponentType.INFERENCE] = PopComponentType.INFERENCE
    inferenceTypes: List[InferenceType] | None = None
    hidden: bool | None = None
    modelUuid: Annotated[str | None, Field(default=None, deprecated='modelUuid is deprecated, use abilityUuid instead'), ]
    model: Annotated[str | None, Field(default=None, deprecated='model is deprecated, use ability instead'), ]
    abilityUuid: str | None = None
    ability: str | None = None
    categoryName: str | None = None
    confidenceThreshold: float | None = None
    topK: int | None = None
    targetFps: str | None = None
    params: dict[str, Any] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')


class MotionModel(enum.StrEnum):
    RANDOM_WALK = "random_walk"
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"


class TrackingComponent(BaseComponent):
    type: Literal[PopComponentType.TRACKING] = PopComponentType.TRACKING
    reidModelUuid: str | None = None
    reidModel: str | None = None
    maxAgeSeconds: float | None = None
    iouThreshold: float | None = None
    simThreshold: float | None = None
    agnostic: bool | None = None
    processNoisePosition: float | None = None
    processNoiseVelocity: float | None = None
    processNoiseAcceleration: float | None = None
    processNoiseScale: float | None = None
    processNoiseAspectRatio: float | None = None
    measurementNoiseCx: float | None = None
    measurementNoiseCy: float | None = None
    measurementNoiseArea: float | None = None
    measurementNoiseAspectRatio: float | None = None
    motionModel: MotionModel | None = None
    downweightLowConfidenceDetections: bool | None = None

    model_config = ConfigDict(extra='forbid')


class ContourType(enum.StrEnum):
    ALL_PIXELS = "all_pixels"
    POLYGON = "polygon"
    CONVEX_HULL = "convex_hull"
    HOUGH_CIRCLES = "hough_circles"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"


class ContourFinderComponent(BaseComponent):
    type: Literal[PopComponentType.CONTOUR_FINDER] = PopComponentType.CONTOUR_FINDER
    contourType: ContourType
    areaThreshold: float | None = None
    model_config = ConfigDict(extra='forbid')


class ComponentFinderComponent(BaseComponent):
    type: Literal[PopComponentType.COMPONENT_FINDER] = PopComponentType.COMPONENT_FINDER
    dilate: float | None = None
    erode: float | None = None
    keepSource: bool | None = None
    componentClassLabel: str | None = None
    model_config = ConfigDict(extra='forbid')


DynamicComponent = Annotated[Union[ForwardComponent | InferenceComponent | TrackingComponent | ContourFinderComponent | ComponentFinderComponent], Field(discriminator="type")]


class Pop(BaseModel):
    components: List[DynamicComponent]
    postTransform: str | None = None
    model_config = ConfigDict(extra='forbid')

# Helper factories

def CropForward(
        targets: List[DynamicComponent],
        maxItems: int | None = None,
        boxPadding: float | None = None,
        orientationTargetAngle: float | None = None,
        includeClasses: list[str] | None = None,
        is_full_fallback: bool = False
) -> PopForward:
    return PopForward(
        operator=PopForwardOperator(
            type=ForwardOperatorType.CROP if not is_full_fallback else ForwardOperatorType.CROP_WITH_FULL_FALLBACK,
            includeClasses=includeClasses,
            crop=PopCrop(
                maxItems=maxItems,
                boxPadding=boxPadding,
                orientationTargetAngle=orientationTargetAngle,
            ),
        ),
        targets=targets
    )

def FullForward(
        targets: List[DynamicComponent],
        includeClasses: list[str] | None = None
) -> PopForward:
    return PopForward(
        operator=PopForwardOperator(
            type=ForwardOperatorType.FULL,
            includeClasses=includeClasses
        ),
        targets=targets,
    )


class ComponentParams(BaseComponent):
    componentId: int
    values: dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
