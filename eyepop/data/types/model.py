"""Model types for the EyePop Data API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from eyepop.data.types.enums import (
    ExportedBy,
    ModelExportFormat,
    ModelExportStatus,
    ModelStatus,
    ModelTask,
    ModelTrainingStage,
    ModelTrainingStatusCode,
    ModelType,
)
from eyepop.data.types.prediction import Prediction


class ModelMetrics(BaseModel):
    cpr: list[tuple[float, float, float]] | None = None


class ModelExport(BaseModel):
    format: ModelExportFormat
    exported_by: ExportedBy
    export_params: dict[str, str] | None = None
    status: ModelExportStatus


ModelExportResponse = ModelExport


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
    name: str | None = None  # type: ignore[assignment]
    description: str | None = None  # type: ignore[assignment]
    is_public: bool | None = None
    external_id: str | None = None
    status: ModelStatus | None = None
    task: ModelTask | None = None
    classes: list[str] | None = None


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


class ModelTrainingEvent(BaseModel):
    """Model training event.

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
    """Internal audit record for model trainings."""

    model_uuid: str
    model: ModelResponse
    work_storage_url: str | None = None
    model_config = ConfigDict(
        protected_namespaces=("pydantic_do_not_prevent_my_prefix_",),
    )


class ExportedUrlResponse(BaseModel):
    model_uuid: str
    model_format: ModelExportFormat
    exported_url: str
    model_config = ConfigDict(
        protected_namespaces=("pydantic_do_not_prevent_my_prefix_",)
    )


class QcAiHubExportParams(BaseModel):
    device_name: str


class Tag(BaseModel):
    name: str
    model_uuid: str
    model_config = ConfigDict(
        protected_namespaces=("pydantic_do_not_prevent_model_prefix_",)
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
        protected_namespaces=("pydantic_do_not_prevent_model_prefix_",)
    )


ExportedAliasResponse = AliasResolution
