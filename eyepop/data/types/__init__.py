"""EyePop Data API types.

This package contains all Pydantic models and enums for the Data API.
Types are organized into domain-specific modules but re-exported here
for backward compatibility.
"""

# Enums and constants
from eyepop.data.types.enums import (
    APPLICATION_JSON,
    MIME_TYPE_APACHE_ARROW_FILE,
    AnnotationInclusionMode,
    AnnotationType,
    ArgoWorkflowPhase,
    ArtifactType,
    AssetInclusionMode,
    AssetStatus,
    AssetUrlType,
    AutoAnnotate,
    AutoAnnotateStatus,
    AutoAnnotateTask,
    ChangeType,
    EvaluationStatus,
    ExportedBy,
    ModelExportFormat,
    ModelExportStatus,
    ModelStatus,
    ModelTask,
    ModelTrainingStage,
    ModelTrainingStatusCode,
    ModelType,
    TranscodeMode,
    UserReview,
    VlmAbilityStatus,
)

# Common types
from eyepop.data.types.common import (
    Box,
    Contour,
    Mask,
    Point2d,
    Point3d,
)

# Prediction types
from eyepop.data.types.prediction import (
    PredictedClass,
    PredictedEmbedding,
    PredictedKeyPoint,
    PredictedKeyPoints,
    PredictedMesh,
    PredictedObject,
    PredictedText,
    Prediction,
)

# Dataset types
from eyepop.data.types.dataset import (
    AutoAnnotateParams,
    AutoAnnotatePrompt,
    Dataset,
    DatasetAutoAnnotate,
    DatasetAutoAnnotateCreate,
    DatasetAutoAnnotateUpdate,
    DatasetCreate,
    DatasetParent,
    DatasetResponse,
    DatasetUpdate,
    DatasetVersion,
    DatasetVersionAssetStats,
    DatasetVersionResponse,
)

# Asset types
from eyepop.data.types.asset import (
    Asset,
    AssetAnnotation,
    AssetAnnotationImport,
    AssetAnnotationResponse,
    AssetImport,
    AssetResponse,
)

# Model types
from eyepop.data.types.model import (
    AliasResolution,
    ExportedAliasResponse,
    ExportedUrlResponse,
    Model,
    ModelAlias,
    ModelAliasCreate,
    ModelAliasResponse,
    ModelAliasUpdate,
    ModelCreate,
    ModelExport,
    ModelExportResponse,
    ModelMetrics,
    ModelResponse,
    ModelSample,
    ModelTrainingAuditRecord,
    ModelTrainingEvent,
    ModelTrainingProgress,
    ModelUpdate,
    QcAiHubExportParams,
    Tag,
    TagResponse,
)

# Event types
from eyepop.data.types.events import (
    ChangeEvent,
    EventHandler,
)

# Workflow types
from eyepop.data.types.workflow import (
    CreateWorkflowBody,
    CreateWorkflowParameters,
    CreateWorkflowResponse,
    DownloadResponse,
    ListWorkflowItem,
    ListWorkflowItemMetadata,
    ListWorkflowItemMetadataLabels,
)

# VLM types
from eyepop.data.types.vlm import (
    AbilityAliasEntry,
    EvaluateFilter,
    EvaluateRequest,
    EvaluateResponse,
    EvaluateRunInfo,
    InferRequest,
    InferRunInfo,
    InferRuntimeConfig,
    TransformInto,
    VlmAbilityCreate,
    VlmAbilityGroupCreate,
    VlmAbilityGroupResponse,
    VlmAbilityGroupUpdate,
    VlmAbilityResponse,
    VlmAbilityUpdate,
)

__all__ = [
    # Enums
    "AnnotationInclusionMode",
    "AnnotationType",
    "ArgoWorkflowPhase",
    "ArtifactType",
    "AssetInclusionMode",
    "AssetStatus",
    "AssetUrlType",
    "AutoAnnotate",
    "AutoAnnotateStatus",
    "AutoAnnotateTask",
    "ChangeType",
    "EvaluationStatus",
    "ExportedBy",
    "ModelExportFormat",
    "ModelExportStatus",
    "ModelStatus",
    "ModelTask",
    "ModelTrainingStage",
    "ModelTrainingStatusCode",
    "ModelType",
    "TranscodeMode",
    "UserReview",
    "VlmAbilityStatus",
    # Constants
    "APPLICATION_JSON",
    "MIME_TYPE_APACHE_ARROW_FILE",
    # Common
    "Box",
    "Contour",
    "Mask",
    "Point2d",
    "Point3d",
    # Prediction
    "PredictedClass",
    "PredictedEmbedding",
    "PredictedKeyPoint",
    "PredictedKeyPoints",
    "PredictedMesh",
    "PredictedObject",
    "PredictedText",
    "Prediction",
    # Dataset
    "AutoAnnotateParams",
    "AutoAnnotatePrompt",
    "Dataset",
    "DatasetAutoAnnotate",
    "DatasetAutoAnnotateCreate",
    "DatasetAutoAnnotateUpdate",
    "DatasetCreate",
    "DatasetParent",
    "DatasetResponse",
    "DatasetUpdate",
    "DatasetVersion",
    "DatasetVersionAssetStats",
    "DatasetVersionResponse",
    # Asset
    "Asset",
    "AssetAnnotation",
    "AssetAnnotationImport",
    "AssetAnnotationResponse",
    "AssetImport",
    "AssetResponse",
    # Model
    "AliasResolution",
    "ExportedAliasResponse",
    "ExportedUrlResponse",
    "Model",
    "ModelAlias",
    "ModelAliasCreate",
    "ModelAliasResponse",
    "ModelAliasUpdate",
    "ModelCreate",
    "ModelExport",
    "ModelExportResponse",
    "ModelMetrics",
    "ModelResponse",
    "ModelSample",
    "ModelTrainingAuditRecord",
    "ModelTrainingEvent",
    "ModelTrainingProgress",
    "ModelUpdate",
    "QcAiHubExportParams",
    "Tag",
    "TagResponse",
    # Events
    "ChangeEvent",
    "EventHandler",
    # Workflow
    "CreateWorkflowBody",
    "CreateWorkflowParameters",
    "CreateWorkflowResponse",
    "DownloadResponse",
    "ListWorkflowItem",
    "ListWorkflowItemMetadata",
    "ListWorkflowItemMetadataLabels",
    # VLM
    "AbilityAliasEntry",
    "EvaluateFilter",
    "EvaluateRequest",
    "EvaluateResponse",
    "EvaluateRunInfo",
    "InferRequest",
    "InferRunInfo",
    "InferRuntimeConfig",
    "TransformInto",
    "VlmAbilityCreate",
    "VlmAbilityGroupCreate",
    "VlmAbilityGroupResponse",
    "VlmAbilityGroupUpdate",
    "VlmAbilityResponse",
    "VlmAbilityUpdate",
]
