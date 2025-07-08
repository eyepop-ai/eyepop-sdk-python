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


class DatasetVersion(BaseModel):
    version: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
    assets_modified_at: datetime | None = None
    last_analysed_at: datetime | None = None
    modifiable: bool = False
    hero_asset_uuid: str | None = None
    asset_stats: DatasetVersionAssetStats | None = None


DatasetVersionResponse = DatasetVersion

class AutoAnnotateTask(enum.StrEnum):
    object_detection = enum.auto()
    image_classification = enum.auto()


class AutoAnnotatePrompt(BaseModel):
    label: str
    prompt: str


class AutoAnnotateParams(BaseModel):
    # @deprecated("candidate_labels is deprecated, use prompts instead")
    candidate_labels: list[str] | None = None
    prompts: list[AutoAnnotatePrompt] | None = None
    confidence_threshold: float | None = None
    task: AutoAnnotateTask = AutoAnnotateTask.object_detection


class DatasetParent(BaseModel):
    dataset_uuid: str
    dataset_version: int


class Dataset(BaseModel):
    uuid: str
    name: str
    description: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    tags: List[str] = []
    account_uuid: str
    auto_annotates: List[AutoAnnotate]
    auto_annotate_params: AutoAnnotateParams | None = None
    versions: List[DatasetVersion] = []
    modifiable_version: int | None = None
    parent: DatasetParent | None = None


DatasetResponse = Dataset


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

class PredictedEmbedding(BaseModel):
    x: float | None = None
    y: float | None = None
    embedding: List[float]

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
    embeddings: List[PredictedEmbedding] | None = None


class AssetAnnotation(BaseModel):
    type: AnnotationType
    user_review: UserReview = UserReview.unknown
    approved_threshold: float | None = None
    auto_annotate: AutoAnnotate | None = None
    auto_annotate_params: AutoAnnotateParams | None = None
    source: str | None = None
    annotation: Prediction | None = None
    uncertainty_score: float | None = None
    source_model_uuid: str | None = None


AssetAnnotationResponse = AssetAnnotation


class Asset(BaseModel):
    uuid: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    mime_type: str | None = None
    file_size_bytes: int | None = None
    status: AssetStatus | None = None
    status_message: str | None = None
    external_id: str | None = None
    partition: str | None = None
    review_priority: float | None = None
    model_relevance: float | None = None
    annotations: List[AssetAnnotation] = []

    ###
    # Denormalized attributes, for convenient serialization/storage detached of a dataset context
    # ###
    dataset_uuid: str | None = None
    account_uuid: str | None = None

    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_model_prefix_',)
    )


AssetResponse = Asset


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
    ModelLess = "ModelLess"


class ModelExportStatus(enum.StrEnum):
    in_progress = enum.auto()
    finished = enum.auto()
    error = enum.auto()


class ExportedBy(enum.StrEnum):
    eyepop = enum.auto()
    qc_ai_hub = enum.auto()


class ModelExport(BaseModel):
    format: ModelExportFormat
    exported_by: ExportedBy
    export_params: dict[str, str] | None = None
    status: ModelExportStatus


ModelExportResponse = ModelExport


class ModelTask(enum.StrEnum):
    object_detection = enum.auto()
    image_classification = enum.auto()


class Model(BaseModel):
    uuid: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    account_uuid: str
    dataset_uuid: str | None = None
    dataset_version: int | None = None
    name: str
    description: str = ""
    type: ModelType | None = None
    is_public: bool = False
    external_id: str | None = None
    pretrained_model_uuid: str | None = None
    extra_params: dict | None = None
    task: ModelTask | None = None
    classes: list[str] | None = None
    status: ModelStatus = ModelStatus.draft
    status_message: str | None = None
    metrics: ModelMetrics | None = None
    exports: list[ModelExport] | None = None


ModelResponse = Model


class ModelCreate(BaseModel):
    name: str
    description: str
    external_id: str | None = None
    pretrained_model_uuid: str | None = None
    extra_params: dict | None = None
    task: ModelTask | None = None
    classes: list[str] | None = None
    type: ModelType | None = None
    status: ModelStatus | None = None


class ModelUpdate(ModelCreate):
    name: str | None = None
    description: str | None = None
    is_public: bool | None = None
    external_id: str | None = None
    status: ModelStatus | None = None
    task: ModelTask | None = None
    classes: list[str] | None = None


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
    started_at: datetime | None = None
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


class Tag(BaseModel):
    name: str
    model_uuid: str
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_model_prefix_',)
    )


TagResponse = Tag


class ModelAlias(BaseModel):
    name: str
    description: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    account_uuid: str
    is_public: bool = False
    tags: list[Tag] = []


ModelAliasResponse = ModelAlias


class ModelAliasCreate(BaseModel):
    name: str
    description: str | None = None
    is_public: bool = False


class ModelAliasUpdate(BaseModel):
    description: str | None = None
    is_public: bool | None = None


class AliasResolution(BaseModel):
    alias: str
    model_uuid: str | None = None
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_model_prefix_',)
    )


ExportedAliasResponse = AliasResolution

MIME_TYPE_APACHE_ARROW_FILE = "application/vnd.apache.arrow.file"


class AssetUrlType(enum.StrEnum):
    gcs = enum.auto()
    s3 = enum.auto()
    https_signed = enum.auto()


class AssetInclusionMode(enum.StrEnum):
    all_assets = enum.auto()
    annotated_only = enum.auto()
    manual_annotated_only = enum.auto()
    auto_annotated_only = enum.auto()


class AnnotationInclusionMode(enum.StrEnum):
    all = enum.auto()
    user_reviewed = enum.auto()
    user_approved = enum.auto()
    ground_truth = enum.auto()


class ModelTrainingStatusCode(enum.Enum):
    ok = 200
    bad_input_error = 400
    internal_error = 500


class ModelTrainingEvent(BaseModel):
    """ Model training event.

        This structure is used for the model training process to communicate the status events.
        The model training workflow SHOULD send at least two updates:

        1. At the beginning of each stage, all attributes except `stage` should be none. The actual
           start time of the stage will be recorded automatically based on the clock time of the receiver.

        2. At the end of each stage, with the same `stage` value and a `status_code`. Any `status_code`
           other than 200 will be interpreted as failure. In this case, the sender should send a huma
           readable, `status_message` for developers to debug the root cause.

        The model training workflow MAY send regular intermediary updates if it can estimate the remaining
        time it will take to complete the current stage. It does so by passing `elapsed_units`,
        `remaining_units_min` as lower bound and optionally `remaining_units_max` as upper bound. The units
        used can be freely chosen by the sender, for example it can be `elapsed seconds`, `used GPU cycles`,
        `epochs`, `batches` or any other unit that is appropriate to express the remaining "work" left,
        relative to the "work" already spend.

        The model training workflow MAY include:
        * `samples` a list of tuples (`asset_uuid`, `prediction`) for a small number of representative
           asset. The prediction are meant to illustrate the current performance of the model in training
           on a set of representative assets.
        * `cpr` A graph described as a list of tuples (`confidence`, `precision`, `recall`) to illustrate
          the current model performance against the validation partition.
        * OR `cpr_transposed` a convenience attribute to send the same data just transposed.
        * `job_identifier` to set an external job identifier if not set already, this will be ignored
          if the external job identifier is already set.
        * `work_storage_url` to set the work storage url if not set already, this will be ignored
          if the work storage url is already set.

    """
    stage: ModelTrainingStage
    job_identifier: str | None = None
    work_storage_url: str | None = None
    elapsed_units: float | None = None
    remaining_units_min: float | None = None
    remaining_units_max: float | None = None
    status_code: ModelTrainingStatusCode | None = None
    status_message: str | None = None
    samples: list[tuple[str, Prediction]] | None = None
    cpr: list[tuple[float, float, float]] | None = None
    cpr_transposed: tuple[list[float], list[float], list[float]] | None = None


class ModelTrainingAuditRecord(BaseModel):
    """ Internal audit record for model trainings.
    """
    model_uuid: str
    model: ModelResponse
    work_storage_url: str | None = None
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_my_prefix_',),
    )


class ExportedUrlResponse(BaseModel):
    model_uuid: str
    model_format: ModelExportFormat
    exported_url: str
    model_config = ConfigDict(
        protected_namespaces=('pydantic_do_not_prevent_my_prefix_',)
    )


class ArtifactType(enum.StrEnum):
    eyepop_bundle = enum.auto()
    weights_file = enum.auto()


class QcAiHubExportParams(BaseModel):
    device_name: str
