import enum
from typing import List, Literal, Annotated, Union, Type

from pydantic import BaseModel, Field


class PopComponentType(enum.StrEnum):
    BASE = "<invalid>"
    FORWARD = "forward"
    INFERENCE = "inference"
    TRACING = "tracing"
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


class PopForwardOperator(BaseModel):
    type: ForwardOperatorType
    includeClasses: List[str] | None = None
    crop: PopCrop | None = None

class PopForward(BaseModel):
    operator: PopForwardOperator | None = None
    targets: List["DynamicComponent"] | None = None

class BaseComponent(BaseModel):
    type: Literal[PopComponentType.BASE] = PopComponentType.BASE
    forward: PopForward | None = None


class ForwardComponent(BaseComponent):
    type: Literal[PopComponentType.FORWARD] = PopComponentType.FORWARD


class InferenceType(enum.StrEnum):
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    KEY_POINTS = "key_points"
    OCR = "ocr"
    MESH = "mesh"
    FEATURE_VECTOR = "feature_vector"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    SEGMENTATION = "segmentation"


class InferenceComponent(BaseComponent):
    type: Literal[PopComponentType.INFERENCE] = PopComponentType.INFERENCE
    inferenceTypes: List[InferenceType]
    hidden: bool | None = None
    modelUuid: str | None = None
    model: str | None = None
    categoryName: str | None = None
    confidenceThreshold: float | None = None
    targetFps: str | None = None


class TracingComponent(BaseComponent):
    type: Literal[PopComponentType.TRACING] = PopComponentType.TRACING
    reidModelUuid: str | None = None
    reidModel: str | None = None
    maxAgeSeconds: float | None = None
    iouThreshold: float | None = None
    simThreshold: float | None = None


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


class ComponentFinderComponent(BaseComponent):
    type: Literal[PopComponentType.COMPONENT_FINDER] = PopComponentType.COMPONENT_FINDER
    dilate: float | None = None
    erode: float | None = None
    keepSource: bool | None = None
    componentClassLabel: str | None = None


DynamicComponent = Annotated[Union[ForwardComponent | InferenceComponent | TracingComponent | ContourFinderComponent | ComponentFinderComponent], Field(discriminator="type")]


class Pop(BaseModel):
    components: List[DynamicComponent]
    postTransform: str | None = None
