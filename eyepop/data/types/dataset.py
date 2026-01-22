from datetime import datetime
from typing import Any, List

from pydantic import BaseModel, ConfigDict

from eyepop.data.types.enums import AutoAnnotate, AutoAnnotateStatus, AutoAnnotateTask


class AutoAnnotatePrompt(BaseModel):
    label: str
    prompt: str


class AutoAnnotateParams(BaseModel):
    # @deprecated("candidate_labels is deprecated, use prompts instead")
    candidate_labels: list[str] | None = None
    prompts: list[AutoAnnotatePrompt] | None = None
    confidence_threshold: float | None = None
    task: AutoAnnotateTask = AutoAnnotateTask.object_detection


class DatasetVersionAssetStats(BaseModel):
    total: int | None = None
    accepted: int | None = None
    rejected: int | None = None
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
    searchable: bool | None = None


DatasetResponse = Dataset


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = []
    auto_annotates: list[AutoAnnotate] = []
    auto_annotate_params: AutoAnnotateParams | None = None
    searchable: bool | None = None
    model_config = ConfigDict(extra="forbid")


class DatasetUpdate(DatasetCreate):
    name: str | None = None  # type: ignore[assignment]
    description: str | None = None  # type: ignore[assignment]
    tags: List[str] | None = None  # type: ignore[assignment]
    auto_annotates: List[AutoAnnotate] | None = None  # type: ignore[assignment]
    auto_annotate_params: AutoAnnotateParams | None = None
    searchable: bool | None = None
    model_config = ConfigDict(extra="forbid")


class DatasetAutoAnnotateCreate(BaseModel):
    auto_annotate: AutoAnnotate
    auto_annotate_params: dict[str, Any] | None = None
    source: str
    source_model_uuid: str | None = None
    status: AutoAnnotateStatus | None = None
    metrics: dict[str, Any] | None = None
    model_config = ConfigDict(extra="forbid")


class DatasetAutoAnnotateUpdate(BaseModel):
    status: AutoAnnotateStatus | None = None
    metrics: dict[str, Any] | None = None
    model_config = ConfigDict(extra="forbid")


class DatasetAutoAnnotate(BaseModel):
    created_at: datetime | None = None
    updated_at: datetime | None = None
    dataset_uuid: str
    dataset_version: int
    source_model_uuid: str | None = None
    auto_annotate: AutoAnnotate | None = None
    auto_annotate_params: dict[str, Any] | None = None
    status: AutoAnnotateStatus | None = None
    status_message: str | None = None
    source: str | None = None
    metrics: dict[str, Any] | None = None
