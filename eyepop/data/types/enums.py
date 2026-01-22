import enum


class AssetStatus(enum.StrEnum):
    rejected = enum.auto()
    upload_in_progress = enum.auto()
    transform_in_progress = enum.auto()
    enrich_in_progress = enum.auto()
    accepted = enum.auto()


class TranscodeMode(enum.StrEnum):
    original = enum.auto()
    normalized = enum.auto()
    video_original_size = enum.auto()
    image_original_size = enum.auto()
    image_fit_1024 = enum.auto()
    image_fit_640 = enum.auto()
    image_fit_224 = enum.auto()
    image_cover_1024 = enum.auto()
    image_cover_640 = enum.auto()
    image_cover_224 = enum.auto()


class ModelType(enum.StrEnum):
    epdet_b1_1 = enum.auto()
    imported = enum.auto()
    vlm_ability = enum.auto()


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


class AutoAnnotateTask(enum.StrEnum):
    object_detection = enum.auto()
    image_classification = enum.auto()


class AutoAnnotateStatus(enum.StrEnum):
    error = enum.auto()
    requested = enum.auto()
    in_progress = enum.auto()
    completed = enum.auto()


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


class ModelTask(enum.StrEnum):
    object_detection = enum.auto()
    image_classification = enum.auto()
    keypoint_detection = enum.auto()


class ModelTrainingStage(enum.StrEnum):
    waiting = enum.auto()
    scheduling = enum.auto()
    preparing = enum.auto()
    training = enum.auto()
    exporting = enum.auto()


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

    workflow_started = enum.auto()
    workflow_succeeded = enum.auto()
    workflow_failed = enum.auto()

    workflow_task_started = enum.auto()
    workflow_task_succeeded = enum.auto()
    workflow_task_failed = enum.auto()


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


class ArgoWorkflowPhase(enum.StrEnum):
    UNKNOWN = "Unknown"
    PENDING = "Pending"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
    ERROR = "Error"


class ArtifactType(enum.StrEnum):
    eyepop_bundle = enum.auto()
    weights_file = enum.auto()


class EvaluationStatus(enum.StrEnum):
    success = "success"
    failed = "failed"


class VlmAbilityStatus(enum.StrEnum):
    draft = "draft"
    published = "published"


# Type aliases
AutoAnnotate = str

# Constants
MIME_TYPE_APACHE_ARROW_FILE = "application/vnd.apache.arrow.file"
APPLICATION_JSON = "application/json"
