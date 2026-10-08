import enum
from typing import Annotated, Any, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from eyepop.data.types.asset import Area
from eyepop.worker.camera import Camera


class PredictionVersion(enum.IntEnum):
    V1 = 1
    V2 = 2
    # V3 adds selected predictions: a select_crop or select_full forward
    # reports each selection on a prediction of its own, `selected` set and
    # timed by the past frame it selected. Nothing else differs from V2.
    V3 = 3


DEFAULT_PREDICTION_VERSION = PredictionVersion.V2


class VideoMode(enum.StrEnum):
    STREAM = "stream"
    BUFFER = "buffer"


class PopComponentType(enum.StrEnum):
    BASE = "<invalid>"
    FORWARD = "forward"
    INFERENCE = "inference"
    TRACKING = "tracking"
    CONTOUR_FINDER = "contour_finder"
    COMPONENT_FINDER = "component_finder"


class ForwardOperatorType(enum.StrEnum):
    FULL = "full"
    CROP = "crop"
    CROP_WITH_FULL_FALLBACK = "crop_with_full_fallback"
    # SELECT_CROP and SELECT_FULL forward one past frame per track, the one of
    # its most relevant detection, and their targets' results arrive late as
    # selected predictions. Only a tracking component's forward can select.
    SELECT_CROP = "select_crop"
    SELECT_FULL = "select_full"


class SelectMode(enum.StrEnum):
    MOST_RELEVANT = "most_relevant"


class PopSelect(BaseModel):
    """How a select_crop or select_full forward picks one frame per track.

    `relevancyModel` (by alias) or `relevancyModelUuid` names an ability that
    runs on every tracked object of every frame; a detection it finds nothing
    on is never selected. Without one, relevance comes from the detection's
    box alone.

    A track shorter than `minTrackLengthSeconds` is never selected. With
    `intervalSeconds` the first selection comes that long after the track
    starts, then at most one per interval and only when a more relevant
    detection turned up; the track's end reports a final one if it improved
    since. Without it, a track is selected once, when it ends.
    """

    mode: SelectMode | None = None
    relevancyModelUuid: str | None = None
    relevancyModel: str | None = None
    minTrackLengthSeconds: float | None = None
    intervalSeconds: float | None = None
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def _validate(self) -> "PopSelect":
        if self.relevancyModel is not None and self.relevancyModelUuid is not None:
            raise ValueError("select can only have one of relevancyModelUuid or relevancyModel")
        if self.minTrackLengthSeconds is not None and self.minTrackLengthSeconds < 0:
            raise ValueError("select minTrackLengthSeconds cannot be negative")
        if self.intervalSeconds is not None and self.intervalSeconds <= 0:
            raise ValueError("select intervalSeconds must be positive")
        return self


class PopCrop(BaseModel):
    maxItems: int | None = None
    boxPadding: float | None = None
    orientationTargetAngle: float | None = None
    model_config = ConfigDict(extra='forbid')


class PopForwardOperator(BaseModel):
    type: ForwardOperatorType
    includeClasses: list[str] | None = None
    crop: PopCrop | None = None
    select: PopSelect | None = None
    model_config = ConfigDict(extra='forbid')

    def is_select(self) -> bool:
        return self.type in (ForwardOperatorType.SELECT_CROP, ForwardOperatorType.SELECT_FULL)

    @model_validator(mode='after')
    def _validate(self) -> "PopForwardOperator":
        # the same rules the worker applies when it compiles the Pop, checked
        # here so a mistake fails where it is made; that a select sits on a
        # tracking component's forward is left to the worker
        if not self.is_select():
            if self.select is not None:
                raise ValueError("select is only valid with the select_crop or select_full operator")
            return self
        if self.select is None:
            raise ValueError(f"{self.type} requires a select block")
        if self.crop is not None:
            if self.type != ForwardOperatorType.SELECT_CROP:
                raise ValueError(f"crop options are only valid with select_crop, not {self.type}")
            if self.crop.maxItems is not None:
                raise ValueError("select_crop forwards one detection per selection, maxItems does not apply")
        return self

class PopForward(BaseModel):
    operator: PopForwardOperator | None = None
    targets: List["DynamicComponent"] | None = None
    model_config = ConfigDict(extra='forbid')

class BaseComponent(BaseModel):
    """The fields every Pop component shares.

    `toWorld` enriches this component's point based predictions with world
    coordinates, back-projected through the Pop's `depthMap`. It is
    declared here because the instance declares it on one shared PopComponent,
    but only a component that runs its own inference can honour it - inference
    and tracking - since that is what gives it an id for the worker to select
    on. Asking for it on a forward, contour finder or component finder is
    rejected when the Pop is compiled. A contour finder's points do get
    enriched, but they belong to the object that fed it, so the request goes on
    the inference component upstream.

    Enrichment needs a *metric* depth ability. A `relative` map is accepted and
    silently produces no world coordinates: its shift is unknown, so a cloud
    recovered from it would be distorted rather than merely unscaled.
    """

    type: Literal[PopComponentType.BASE] = PopComponentType.BASE
    id: int | None = None
    forward: PopForward | None = None
    toWorld: bool | None = None
    model_config = ConfigDict(extra='forbid')


# Each component narrows `type` to its own literal, which is what makes the
# DynamicComponent union discriminated. basedpyright reports every one of these
# as an incompatible override because a model field is mutable and so invariant;
# the narrowing is the whole point of the pattern, so it is suppressed per line
# rather than by turning the rule off across the package.
class ForwardComponent(BaseComponent):
    type: Literal[PopComponentType.FORWARD] = PopComponentType.FORWARD  # pyright: ignore[reportIncompatibleVariableOverride]
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
    type: Literal[PopComponentType.INFERENCE] = PopComponentType.INFERENCE  # pyright: ignore[reportIncompatibleVariableOverride]
    inferenceTypes: List[InferenceType] | None = None
    hidden: bool | None = None
    modelUuid: Annotated[str | None, Field(default=None, deprecated='modelUuid is deprecated, use abilityUuid instead'), ]
    model: Annotated[str | None, Field(default=None, deprecated='model is deprecated, use ability instead'), ]
    abilityUuid: str | None = None
    ability: str | None = None
    categoryName: str | None = None
    confidenceThreshold: float | None = None
    objectAreaThreshold: float | None = None
    topK: int | None = None
    topKClasses: int | None = None
    targetFps: str | None = None
    videoChunkLengthSeconds: float | None = None
    videoChunkOverlap: float | None = None
    params: dict[str, Any] | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')


class MotionModel(enum.StrEnum):
    RANDOM_WALK = "random_walk"
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"


class TrackingComponent(BaseComponent):
    type: Literal[PopComponentType.TRACKING] = PopComponentType.TRACKING  # pyright: ignore[reportIncompatibleVariableOverride]
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
    classBeta: float | None = None
    classGamma: float | None = None
    classHysteresis: bool | None = None
    classHysteresisHighThreshold: float | None = None
    classHysteresisLowThreshold: float | None = None
    classHysteresisMinHoldFrames: int | None = None
    classHysteresisAllowedClasses: list[str] | None = None

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
    type: Literal[PopComponentType.CONTOUR_FINDER] = PopComponentType.CONTOUR_FINDER  # pyright: ignore[reportIncompatibleVariableOverride]
    # Left unset the worker fits a polygon. Defaulted to None rather than to
    # POLYGON so the default stays the worker's to change, the way every other
    # optional component attribute here works.
    contourType: ContourType | None = None
    areaThreshold: float | None = None
    model_config = ConfigDict(extra='forbid')


class ComponentFinderComponent(BaseComponent):
    type: Literal[PopComponentType.COMPONENT_FINDER] = PopComponentType.COMPONENT_FINDER  # pyright: ignore[reportIncompatibleVariableOverride]
    dilate: float | None = None
    erode: float | None = None
    keepSource: bool | None = None
    componentClassLabel: str | None = None
    model_config = ConfigDict(extra='forbid')


DynamicComponent = Annotated[Union[ForwardComponent | InferenceComponent | TrackingComponent | ContourFinderComponent | ComponentFinderComponent], Field(discriminator="type")]


class SourceDefaults(BaseModel):
    """Source level parameters a Pop sets once for every source it processes.

    Scoped to what describes the scene and the capture, where one setting
    sensibly covers every source of a Pop. Identity and transport - source id,
    url, whether it is live - are deliberately absent, since defaulting those is
    meaningless.

    Merged per field rather than per block: a source giving its own roi but no
    camera keeps its roi and takes the default camera. Anything the source sets
    wins. Because the camera merges as one field, a source declaring its own
    lens replaces a defaulted one outright rather than mixing the two.
    """

    camera: Camera | None = None
    roi: Area | None = None
    fps: str | None = None
    motionDetect: bool | None = None
    motionSensitivity: float | None = None
    motionThreshold: float | None = None
    motionGap: int | None = None
    motionGridX: int | None = None
    motionGridY: int | None = None
    model_config = ConfigDict(extra='forbid')


class PopDepthMap(BaseModel):
    """The depth ability whose frame level map feeds world coordinates.

    `ability` names it by alias and `abilityUuid` by uuid; give exactly one.
    A Pop has one depth map because the worker back-projects every prediction
    through it, so a second depth source would have nowhere to go. Naming one
    makes the converter build the depth branch itself and keep it out of the
    response - the caller asked for coordinates, not for a megabyte of base64
    depth per frame.

    `toWorld` back-projects the map itself, so the response carries a point
    cloud of the whole scene rather than one per segmented object. It is also
    what reveals the map: without it the injected branch stays out of the
    response entirely. Read the result with `eyepop.PointCloud.from_depth`.

    Use a *metric* depth ability. A `relative` one is accepted and yields no
    world coordinates at all.
    """

    ability: str | None = None
    abilityUuid: str | None = None
    toWorld: bool | None = None
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def _validate(self) -> "PopDepthMap":
        # checked here rather than left to the worker, which rejects both cases:
        # naming no ability is a depth branch that cannot be built, and naming
        # two is a Pop with no right answer
        if (self.ability is None) == (self.abilityUuid is None):
            raise ValueError("depthMap requires exactly one of ability or abilityUuid")
        return self


class Pop(BaseModel):
    """A Pop: the components to run and how to run them."""

    components: List[DynamicComponent]
    postTransform: str | None = None
    defaults: SourceDefaults | None = None
    depthMap: PopDepthMap | None = None
    model_config = ConfigDict(extra='forbid')

    def selects(self) -> bool:
        """Whether any forward of this Pop is a select_crop or select_full."""
        return _any_selects(self.components)


def _any_selects(components: List[DynamicComponent] | None) -> bool:
    for component in components or []:
        forward = component.forward
        if forward is None:
            continue
        if forward.operator is not None and forward.operator.is_select():
            return True
        if _any_selects(forward.targets):
            return True
    return False

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


def SelectForward(
        targets: List[DynamicComponent],
        full: bool = False,
        relevancyModel: str | None = None,
        relevancyModelUuid: str | None = None,
        minTrackLengthSeconds: float | None = None,
        intervalSeconds: float | None = None,
        boxPadding: float | None = None,
        orientationTargetAngle: float | None = None,
) -> PopForward:
    """Forwards each track's most relevant detection to `targets`, late.

    Goes on a tracking component. The targets run on a crop of that detection
    from the past frame it was seen in, or with `full` on that whole frame.
    Their results arrive as selected predictions: `Prediction.selected` is
    set, `timestamp` is the past frame's and the selected object's `trackId`
    links them to the live predictions. `boxPadding` and
    `orientationTargetAngle` shape the crop and do not apply with `full`.
    """
    crop = None
    if boxPadding is not None or orientationTargetAngle is not None:
        crop = PopCrop(boxPadding=boxPadding, orientationTargetAngle=orientationTargetAngle)
    return PopForward(
        operator=PopForwardOperator(
            type=ForwardOperatorType.SELECT_FULL if full else ForwardOperatorType.SELECT_CROP,
            crop=crop,
            select=PopSelect(
                mode=SelectMode.MOST_RELEVANT,
                relevancyModel=relevancyModel,
                relevancyModelUuid=relevancyModelUuid,
                minTrackLengthSeconds=minTrackLengthSeconds,
                intervalSeconds=intervalSeconds,
            ),
        ),
        targets=targets,
    )


class ComponentParams(BaseModel):
    componentId: int
    values: dict[str, Any]
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')


class MotionDetectConfig(BaseModel):
    motionDetect: bool = Field(description="Whether or not to pause processing based on motion detection", default=True)
    motionSensitivity: float | None = Field(description="Sensitivity of motion detection as percentage of pixels that must not change to not detect a motion in that cell, default is 0.5 ", default=None)
    motionThreshold: float | None = Field(description="Threshold percentage of cells that must change for a motion event to trigger, default is 0.01", default=None)
    motionGap: int | None = Field(description="Gap of no detected motion in seconds before motion-stopped event is trigger, default is 5", default=None)
    motionGridX: int | None = Field(description="Grid x size of motion detection grid, default is 10", default=None)
    motionGridY: int | None = Field(description="Grid y size of motion detection grid, default is 10", default=None)
