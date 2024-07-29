import enum
from datetime import datetime
from typing import List, Optional, Type

from pydantic import BaseModel


class AssetStatus(enum.StrEnum):
    rejected = enum.auto()
    upload_in_progress = enum.auto()
    transform_in_progress = enum.auto()
    enrich_in_progress = enum.auto()
    accepted = enum.auto()


class TranscodeMode(enum.StrEnum):
    original = enum.auto()
    image_original_size = enum.auto()
    image_fit_1024 = enum.auto()
    image_fit_640 = enum.auto()
    image_fit_224 = enum.auto()
    image_cover_1024 = enum.auto()
    image_cover_640 = enum.auto()
    image_cover_224 = enum.auto()


AutoAnnotate = str


class ModelType(enum.StrEnum):
    epdet_b1 = enum.auto()


class ModelStatus(enum.StrEnum):
    error = enum.auto()
    draft = enum.auto()
    requested = enum.auto()
    in_progress = enum.auto()
    available = enum.auto()
    published = enum.auto()


class AnnotationType(enum.StrEnum):
    manual = enum.auto()
    prediction = enum.auto()
    auto = enum.auto()


class UserReview(enum.StrEnum):
    approved = enum.auto()
    rejected = enum.auto()
    unknown = enum.auto()


class DatasetVersionResponse(BaseModel):
    version: int
    created_at: datetime
    updated_at: datetime
    asset_count: int
    modifiable: bool
    hero_asset_uuid: Optional[str] = None


class DatasetResponse(BaseModel):
    uuid: str
    name: str
    description: Optional[str] = ""
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    account_uuid: str
    auto_annotates: List[AutoAnnotate]
    versions: List[DatasetVersionResponse]


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    tags: List[str] = []
    auto_annotates: List[AutoAnnotate] = []


class DatasetUpdate(DatasetCreate):
    pass


class Point2d(BaseModel):
    x: float
    y: float


class Point3d(Point2d):
    z: Optional[float] = None


class Box(BaseModel):
    topLeft: Point2d
    bottomRight: Point2d


class PredictedClass(BaseModel):
    id: Optional[int] = None
    confidence: Optional[float] = None
    classLabel: str
    category: Optional[str] = None


class PredictedLabel(BaseModel):
    id: Optional[int] = None
    confidence: Optional[float] = None
    label: str
    category: Optional[str] = None


class Contour(BaseModel):
    points: List[Point2d]
    cutouts: List[List[Point2d]]


class Mask(BaseModel):
    bitmap: str
    width: int
    height: int
    stride: int


class PredictedMesh(BaseModel):
    id: Optional[int] = None
    category: Optional[str] = None
    confidence: Optional[float] = None
    points: List[Point3d]


class PredictedKeyPoint(Point3d, PredictedClass):
    visible: Optional[bool] = None


class PredictedKeyPoints(BaseModel):
    category: Optional[str] = None
    type: Optional[str] = None
    points: List[PredictedKeyPoint]


class PredictedObject(PredictedClass):
    traceId: Optional[int] = None
    x: float
    y: float
    width: float
    height: float
    orientation: Optional[float] = None
    outline: Optional[List[Point2d]] = None
    contours: Optional[List[Contour]] = None
    mask: Optional[Mask] = None
    objects: Optional[List["PredictedObject"]] = None
    classes: Optional[List[PredictedClass]] = None
    labels: Optional[List[PredictedLabel]] = None
    meshs: Optional[List[PredictedMesh]] = None
    keyPoints: Optional[List[PredictedKeyPoints]] = None


class Prediction(BaseModel):
    source_width: float
    source_height: float
    objects: Optional[List[PredictedObject]] = None
    classes: Optional[List[PredictedClass]] = None
    labels: Optional[List[PredictedLabel]] = None
    meshs: Optional[List[PredictedMesh]] = None
    keyPoints: Optional[List[PredictedKeyPoints]] = None


class AssetAnnotationResponse(BaseModel):
    type: AnnotationType
    user_review: UserReview
    auto_annotate: Optional[AutoAnnotate] = []
    annotation: Optional[Prediction] = None
    uncertainty_score: Optional[float] = None


class AssetResponse(BaseModel):
    uuid: str
    created_at: datetime
    updated_at: datetime
    mime_type: str
    file_size_bytes: Optional[int] = None
    status: Optional[AssetStatus] = None
    status_message: Optional[str] = None
    external_id: Optional[str] = None
    annotations: Optional[List[AssetAnnotationResponse]] = []


class AssetImport(BaseModel):
    url: str
    manual_annotation: Optional[Prediction] = None


class ModelResponse(BaseModel):
    uuid: str
    created_at: datetime
    updated_at: datetime
    account_uuid: str
    dataset_uuid: str
    dataset_version: int
    name: str
    description: Optional[str] = None
    type: ModelType
    status: ModelStatus
    status_message: Optional[str] = None
    exported_url: Optional[str] = None


class ModelCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: ModelType


class ModelUpdate(ModelCreate):
    pass