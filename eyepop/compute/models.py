import os
from enum import Enum

from pydantic import BaseModel, Field


class ComputeContext(BaseModel):
    compute_url: str = Field(description="The url of the compute api", default=os.getenv("_COMPUTE_API_URL", "https://compute.staging.eyepop.xyz"))
    session_endpoint: str = Field(description="The endpoint of the session", default="")
    session_uuid: str = Field(description="The uuid of the session", default="")
    pipeline_uuid: str = Field(description="The uuid of the pipeline", default="")
    pipeline_id: str = Field(description="The id of the pipeline", default="")
    user_uuid: str = Field(description="The uuid of the user", default=os.getenv("EYEPOP_USER_UUID", ""))
    secret_key: str = Field(description="The secret key of the user", default=os.getenv("EYEPOP_SECRET_KEY", ""))
    access_token: str = Field(description="The JWT access token from compute API", default="")
    wait_for_session_timeout: int = Field(description="The timeout for the session", default=60)
    wait_for_session_interval: int = Field(description="The interval for the session", default=2)

class PipelineStatus(str, Enum):
    UNKNOWN = "unknown"
    PENDING = "pending"
    PIPELINE_CREATING = "pipeline_creating"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ERROR = "error"
    
    @classmethod
    def _missing_(cls, value):
        """Handle unknown status values by checking if they contain known keywords."""
        if isinstance(value, str):
            value_lower = value.lower()
            # Check for keywords in the status string (order matters - check more specific first)
            if "error" in value_lower:
                return cls.ERROR
            elif "fail" in value_lower:
                return cls.FAILED
            elif "stop" in value_lower:
                return cls.STOPPED
            elif "running" in value_lower or "run" in value_lower:
                return cls.RUNNING
            elif "pending" in value_lower or "creat" in value_lower or "start" in value_lower:
                return cls.PENDING
        # If no match, return UNKNOWN instead of raising an error
        return cls.UNKNOWN


class ComputeApiPipelineStatus(BaseModel):
    status: PipelineStatus = Field(description="The status of the pipeline")
    reason: str = Field(description="The reason for the status")


class ComputeApiSessionRequest(BaseModel):
    account_uuid: str = Field(description="Required account uuid to create a session")


class ComputeApiSessionResponse(BaseModel):
    session_uuid: str = Field(description="The related session uuid for this session")
    session_endpoint: str = Field(description="The related session url for this session")
    access_token: str = Field(description="The JWT access token for session authentication")
    pipelines: list = Field(description="List of pipelines in the session", default=[])
    pipeline_uuid: str = Field(description="The related pipeline uuid for this session", default="")
    pipeline_version: str = Field(description="The related pipeline version for this session", default="")
    session_status: PipelineStatus = Field(description="The status of the session", default=PipelineStatus.PENDING)
    session_message: str = Field(description="The message of the session", default="")
    pipeline_ttl: int | None = Field(description="The ttl of the pipeline", default=None)
    session_active: bool = Field(description="Whether the session is active", default=False)
