"""Workflow types for the EyePop Data API."""

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel

from eyepop.data.types.enums import ArgoWorkflowPhase, AssetUrlType


class CreateWorkflowBody(BaseModel):
    parameters: dict | None = None


class CreateWorkflowParameters(BaseModel):
    dataset_uuid: str | None = None
    dataset_version: int | None = None
    model_uuid: str | None = None
    config: Dict[str, Any] | None = None
    root_base_url: str | None = None


class CreateWorkflowResponse(BaseModel):
    workflow_id: str


class ListWorkflowItemMetadataLabels(BaseModel):
    account_uuid: str
    dataset_uuid: str | None = None
    model_uuid: str | None = None
    phase: ArgoWorkflowPhase


class ListWorkflowItemMetadata(BaseModel):
    workflow_id: str
    created_at: datetime
    labels: ListWorkflowItemMetadataLabels


class ListWorkflowItem(BaseModel):
    metadata: ListWorkflowItemMetadata


class DownloadResponse(BaseModel):
    url: str
    url_type: AssetUrlType
