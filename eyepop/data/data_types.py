import enum
from datetime import datetime
from typing import List, Callable, Awaitable

from pydantic import BaseModel, ConfigDict


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
    epdet_b1_1 = enum.auto()
    yolov7 = enum.auto()
    yolov7_tiny = enum.auto()
    yolov7_e6e = enum.auto()
    yolov7_w6 = enum.auto()
    yolov7_x = enum.auto()
    coolr_demo = enum.auto()
    imported = enum.auto()
    eyepop_sst = enum.auto()


class ModelStatus(enum.StrEnum):
    error = enum.auto()
    draft = enum.auto()
    requested = enum.auto()
    in_progress = enum.auto()
    available = enum.auto()
    published = enum.auto()


class AnnotationType(enum.StrEnum):
    ground_truth = enum.auto()
    prediction = enum.auto()
    auto = enum.auto()


class UserReview(enum.StrEnum):
    approved = enum.auto()
    rejected = enum.auto()
    unknown = enum.auto()


class DatasetVersionAssetStats(BaseModel):
    total: int | None = None
    accepted: int | None = None
    annotated: int | None = None
    auto_annotated: int | None = None
    auto_annotated_approved: int | None = None
    ground_truth_annotated: int | None = None


class DatasetVersionResponse(BaseModel):
    version: int
    created_at: datetime
    updated_at: datetime
    assets_modified_at: datetime
    last_analysed_at: datetime | None = None
    modifiable: bool
    hero_asset_uuid: str | None = None
    asset_stats: DatasetVersionAssetStats | None = None


class AutoAnnotateParams(BaseModel):
    candidate_labels: list[str] | None = None
    prompt: str | None = None
    confidence_threshold: float | None = None


class DatasetResponse(BaseModel):
    uuid: str
    name: str
    description: str = ""
    created_at: datetime
    updated_at: datetime | None = None
    tags: List[str]
    account_uuid: str
    auto_annotates: List[AutoAnnotate]
    auto_annotate_params: AutoAnnotateParams | None = None
    versions: List[DatasetVersionResponse]
    modifiable_version: int | None = None


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = []
    auto_annotates: list[AutoAnnotate] = []
    auto_annotate_params: AutoAnnotateParams | None = None


class DatasetUpdate(DatasetCreate):
    name: str | None = None
    description: str | None = None
    tags: List[str] | None = None
    auto_annotates: List[AutoAnnotate]  | None = None
    auto_annotate_params: AutoAnnotateParams | None = None


class Point2d(BaseModel):
    x: float
    y: float


class Point3d(Point2d):
    z: float | None = None


class Box(BaseModel):
    topLeft: Point2d
    bottomRight: Point2d


class PredictedClass(BaseModel):
    id: int | None = None
    confidence: float | None = None
    classLabel: str
    category: str | None = None


class PredictedText(BaseModel):
    id: int | None = None
    confidence: float | None = None
    text: str
    category: str | None = None


class Contour(BaseModel):
    points: List[Point2d]
    cutouts: List[List[Point2d]]


class Mask(BaseModel):
    bitmap: str
    width: int
    height: int
    stride: int


class PredictedMesh(BaseModel):
    id: int | None = None
    category: str | None = None
    confidence: float = None
    points: List[Point3d]


class PredictedKeyPoint(Point3d, PredictedClass):
    visible: bool | None = None


class PredictedKeyPoints(BaseModel):
    category: str | None = None
    type: str | None = None
    points: List[PredictedKeyPoint]


class PredictedObject(PredictedClass):
    traceId: int | None = None
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
    source_width: float
    source_height: float
    objects: List[PredictedObject] | None = None
    classes: List[PredictedClass] | None = None
    texts: List[PredictedText] | None = None
    meshs: List[PredictedMesh] | None = None
    keyPoints: List[PredictedKeyPoints] | None = None


class AssetAnnotationResponse(BaseModel):
    type: AnnotationType
    user_review: UserReview
    approved_threshold: float | None = None
    auto_annotate: AutoAnnotate | None = None
    auto_annotate_params: AutoAnnotateParams | None = None
    source: str | None = None
    annotation: Prediction | None = None
    uncertainty_score: float | None = None
    source_model_uuid: str | None = None

    class Config:
        use_enum_values = True


class AssetResponse(BaseModel):
    uuid: str
    created_at: datetime
    updated_at: datetime
    mime_type: str
    file_size_bytes: int | None = None
    status: AssetStatus | None = None
    status_message: str | None = None
    external_id: str | None = None
    partition: str | None = None
    review_priority: float | None = None
    model_relevance: float | None = None
    annotations: List[AssetAnnotationResponse] = []
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_model_prefix_',)
    )


class AssetImport(BaseModel):
    url: str
    ground_truth: Prediction | None = None


class ModelMetrics(BaseModel):
    cpr: list[tuple[float, float, float]] | None = None


class ModelExportFormat(enum.StrEnum):
    TensorFlowLite = "TensorFlowLite"
    TensorFlowGraphDef = "TensorFlowGraphDef"
    TorchScript = "TorchScript"
    TorchScriptCpu = "TorchScriptCpu"
    TorchScriptCuda = "TorchScriptCuda"
    ONNX = "ONNX"
    PyTorch = "PyTorch"


class ModelExportStatus(enum.StrEnum):
    in_progress = enum.auto()
    finished = enum.auto()
    error = enum.auto()


class ExportedBy(enum.StrEnum):
    eyepop = enum.auto()
    qc_ai_hub = enum.auto()


class ModelExportResponse(BaseModel):
    format: ModelExportFormat
    exported_by: ExportedBy
    export_params: dict[str, str] | None = None
    status: ModelExportStatus


class ModelResponse(BaseModel):
    uuid: str
    created_at: datetime
    updated_at: datetime | None = None
    account_uuid: str
    dataset_uuid: str | None = None
    dataset_version: int | None = None
    name: str
    description: str
    type: ModelType | None = None
    status: ModelStatus
    is_public: bool
    external_id: str | None = None
    status_message: str | None = None
    metrics: ModelMetrics | None = None
    exports: list[ModelExportResponse] | None = None


class ModelCreate(BaseModel):
    name: str
    description: str
    external_id: str | None = None


class ModelUpdate(ModelCreate):
    name: str | None = None
    description: str | None = None
    is_public: bool | None = None
    external_id: str | None = None
    status: ModelStatus | None = None


class ModelTrainingStage(enum.StrEnum):
    waiting = enum.auto()
    scheduling = enum.auto()
    preparing = enum.auto()
    training = enum.auto()
    exporting = enum.auto()


class ModelSample(BaseModel):
    asset_uuid: str
    prediction: Prediction


class ModelTrainingProgress(BaseModel):
    stage: ModelTrainingStage
    queue_length: int | None = None
    started_at: datetime
    finished_at: datetime | None = None
    status_code: int | None = None
    status_message: str | None = None
    metrics: ModelMetrics | None = None
    samples: list[ModelSample] | None = None
    remaining_seconds_min: float | None = None
    remaining_seconds_max: float | None = None


class ChangeType(enum.StrEnum):
    dataset_added = enum.auto()
    dataset_removed = enum.auto()
    dataset_modified = enum.auto()
    dataset_version_modified = enum.auto()
    asset_added = enum.auto()
    asset_removed = enum.auto()
    asset_status_modified = enum.auto()
    asset_annotation_modified = enum.auto()
    model_added = enum.auto()
    model_removed = enum.auto()
    model_modified = enum.auto()
    model_status_modified = enum.auto()
    model_progress = enum.auto()
    events_lost = enum.auto()


class ChangeEvent(BaseModel):
    change_type: ChangeType
    account_uuid: str
    dataset_uuid: str | None
    dataset_version: int | None
    asset_uuid: str | None
    mdl_uuid: str | None



EventHandler = Callable[[ChangeEvent], Awaitable[None]]


class TagResponse(BaseModel):
    name: str
    model_uuid: str
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_model_prefix_',)
    )


class ModelAliasResponse(BaseModel):
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    account_uuid: str
    is_public: bool
    tags: list[TagResponse]


class ModelAliasCreate(BaseModel):
    name: str
    description: str | None = None
    is_public: bool = False


class ModelAliasUpdate(BaseModel):
    description: str | None = None
    is_public: bool | None = None


class ModelExportFormat(enum.StrEnum):
    TensorFlowLite = "TensorFlowLite"
    TensorFlowGraphDef = "TensorFlowGraphDef"
    TorchScript = "TorchScript"
    TorchScriptCpu = "TorchScriptCpu"
    TorchScriptCuda = "TorchScriptCuda"
    ONNX = "ONNX"


class ExportedAliasResponse(BaseModel):
    alias: str
    model_uuid: str | None
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_model_prefix_',)
    )
