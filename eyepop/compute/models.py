from pydantic import BaseModel, Field
from pyparsing import Enum


class PipelineStatus(str, Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


class ComputeApiPipelineStatus(BaseModel):
    status: PipelineStatus = Field(description="The status of the pipeline")
    reason: str = Field(description="The reason for the status")


class ComputeApiSessionRequest(BaseModel):
    account_uuid: str = Field(description="Required account uuid to create a session")


class ComputeApiSessionResponse(BaseModel):
    pipeline_url: str = Field(description="The related pipeline url for this session")
    session_name: str = Field(description="The name of the session")
    session_endpoint: str = Field(description="The endpoint of the session")
    user_uuid: str = Field(description="The uuid of the user")
    pipeline_version: str = Field(description="The version of the pipeline")
    pipeline_ttl: str = Field(description="The ttl of the pipeline")
    pipeline_uuid: str = Field(description="The related pipeline uuid for this session")
    session_uuid: str = Field(description="The related session uuid for this session")
    status: PipelineStatus = Field(description="The status of the session")
