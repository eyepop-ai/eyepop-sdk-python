from pydantic import BaseModel, Field
from pyparsing import Enum
import os

class ComputeContext(BaseModel):
    compute_url: str = Field(description="The url of the compute api", default=os.getenv("_COMPUTE_API_URL", "https://compute-api.staging.eyepop.xyz"))
    session_endpoint: str = Field(description="The endpoint of the session", default="")
    session_uuid: str = Field(description="The uuid of the session", default="")
    pipeline_uuid: str = Field(description="The uuid of the pipeline", default="")
    user_uuid: str = Field(description="The uuid of the user", default=os.getenv("EYEPOP_USER_UUID", ""))
    secret_key: str = Field(description="The secret key of the user", default=os.getenv("EYEPOP_SECRET_KEY", ""))
    wait_for_session_timeout: int = Field(description="The timeout for the session", default=10)
    wait_for_session_interval: int = Field(description="The interval for the session", default=1)

class PipelineStatus(str, Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ERROR = "error"


class ComputeApiPipelineStatus(BaseModel):
    status: PipelineStatus = Field(description="The status of the pipeline")
    reason: str = Field(description="The reason for the status")


class ComputeApiSessionRequest(BaseModel):
    account_uuid: str = Field(description="Required account uuid to create a session")


class ComputeApiSessionResponse(BaseModel):
    session_uuid: str = Field(description="The related session uuid for this session")
    session_endpoint: str = Field(description="The related session url for this session")
    pipeline_uuid: str = Field(description="The related pipeline uuid for this session")
    pipeline_version: str = Field(description="The related pipeline version for this session")
    session_status: PipelineStatus = Field(description="The status of the session", default=PipelineStatus.PENDING)
    session_message: str = Field(description="The message of the session", default="")
    pipeline_ttl: int | None = Field(description="The ttl of the pipeline")
    session_active: bool = Field(description="Whether the session is active")
