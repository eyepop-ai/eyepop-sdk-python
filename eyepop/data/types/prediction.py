from typing import List

from pydantic import BaseModel, Field

from eyepop.data.types.common import Contour, Mask, Point2d, Point3d


class PredictedClass(BaseModel):
    id: int | None = None
    confidence: float | None = None
    classLabel: str
    category: str | None = None


class PredictedEmbedding(BaseModel):
    x: float | None = None
    y: float | None = None
    embedding: List[float]
    category: str | None = None


class PredictedText(BaseModel):
    id: int | None = None
    confidence: float | None = None
    text: str
    category: str | None = None


class PredictedMesh(BaseModel):
    id: int | None = None
    category: str | None = None
    confidence: float = None  # type: ignore[assignment]
    points: List[Point3d]


class PredictedKeyPoint(Point3d):
    id: int | None = None
    confidence: float | None = None
    classLabel: str | None = None
    category: str | None = None
    visible: bool | None = None


class PredictedKeyPoints(BaseModel):
    category: str | None = None
    type: str | None = None
    points: List[PredictedKeyPoint]


class PredictedObject(PredictedClass):
    trackId: int | None = None
    x: float
    y: float
    width: float
    height: float
    orientation: float | None = None
    outline: List[Point2d] | None = None
    contours: List[Contour] | None = None
    mask: Mask | None = None
    objects: List["PredictedObject"] | None = None
    classes: List[PredictedClass] | None = None
    texts: List[PredictedText] | None = None
    meshs: List[PredictedMesh] | None = None
    keyPoints: List[PredictedKeyPoints] | None = None


class Prediction(BaseModel):
    """Represents a prediction for an asset or a chunk of an asset."""

    source_width: float = Field(
        description="The width of the source coordinate system for all prediction coordinates."
    )
    source_height: float = Field(
        description="The height of the source coordinate system for all prediction coordinates."
    )
    timestamp: int | None = Field(
        default=None,
        description="Temporal offset of prediction from start of the media (video or audio) in nano seconds.",
    )
    captured_at: int | None = Field(
        default=None,
        description="Real time when the media was captured as epoch timestamp in nano seconds. "
        "Only provided if source provides this timestamp e.g. as timestamp/x-ntp in RTSP.",
    )
    duration: int | None = Field(
        default=None,
        description="Temporal length of the chunk that was the source for this prediction in nano seconds.",
    )
    offset: int | None = Field(
        default=None,
        description="A media specific offset. For video frames, this is the frame number of prediction. "
        "For audio samples, this is the offset of the first sample for this prediction.",
    )
    offset_duration: int | None = Field(
        default=None,
        description="Offset length of the chunk used for this prediction. It has the same format as offset.",
    )

    objects: List[PredictedObject] | None = None
    classes: List[PredictedClass] | None = None
    texts: List[PredictedText] | None = None
    meshs: List[PredictedMesh] | None = None
    keyPoints: List[PredictedKeyPoints] | None = None
    embeddings: List[PredictedEmbedding] | None = None
