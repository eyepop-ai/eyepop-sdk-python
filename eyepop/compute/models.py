import os
from enum import Enum

from pydantic import BaseModel, Field

from eyepop.settings import settings


class ComputeContext(BaseModel):
    """
    Context for Compute API operations.

    Contains session details, authentication tokens, and configuration.
    """
    compute_url: str = Field(
        description="The url of the compute api",
        default_factory=lambda: os.getenv("EYEPOP_URL", settings.default_compute_url)
    )
    session_endpoint: str = Field(description="The endpoint of the session", default="")
    session_uuid: str = Field(description="The uuid of the session", default="")
    pipeline_uuid: str = Field(description="The uuid of the pipeline", default="")
    pipeline_id: str = Field(description="The id of the pipeline", default="")
    user_uuid: str = Field(description="The uuid of the user", default=os.getenv("EYEPOP_USER_UUID", ""))
    api_key: str = Field(description="The api key of the user", default=os.getenv("EYEPOP_API_KEY", ""))
    access_token: str = Field(description="The JWT access token from compute API", default="")
    access_token_expires_at: str = Field(description="ISO timestamp when access token expires", default="")
    access_token_expires_in: int = Field(description="Seconds until access token expires", default=0)
    wait_for_session_timeout: int = Field(
        description="The timeout for the session",
        default_factory=lambda: settings.session_timeout
    )
    wait_for_session_interval: int = Field(
        description="The interval for the session",
        default_factory=lambda: settings.session_interval
    )


class PipelineStatus(str, Enum):
    """
    Possible states for a compute API session/pipeline.
    """
    UNKNOWN = "unknown"
    PENDING = "pending"
    PIPELINE_CREATING = "pipeline_creating"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ERROR = "error"

    @classmethod
    def _missing_(cls, value):
        """
        Handle unknown status values by checking if they contain known keywords.

        This makes the enum more resilient to API changes.
        """
        if isinstance(value, str):
            value_lower = value.lower()
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
        return cls.UNKNOWN


class ComputeApiPipelineStatus(BaseModel):
    """Status information for a pipeline."""
    status: PipelineStatus = Field(description="The status of the pipeline")
    reason: str = Field(description="The reason for the status")


class ComputeApiSessionRequest(BaseModel):
    """Request body for creating a new session."""
    account_uuid: str = Field(description="Required account uuid to create a session")


class ComputeApiSessionResponse(BaseModel):
    """Response from compute API session endpoints."""
    session_uuid: str = Field(description="The related session uuid for this session")
    session_endpoint: str = Field(description="The related session url for this session")
    access_token: str = Field(description="The JWT access token for session authentication")
    access_token_expires_at: str = Field(description="ISO timestamp when access token expires", default="")
    access_token_expires_in: int = Field(description="Seconds until access token expires", default=0)
    access_token_valid_until: int | None = Field(description="The timestamp of the access token valid until", default=None)
    pipelines: list = Field(description="List of pipelines in the session", default=[])
    pipeline_uuid: str = Field(description="The related pipeline uuid for this session", default="")
    pipeline_version: str = Field(description="The related pipeline version for this session", default="")
    session_status: PipelineStatus = Field(description="The status of the session", default=PipelineStatus.PENDING)
    session_message: str = Field(description="The message of the session", default="")
    session_name: str = Field(description="The name of the session", default="")
    user_uuid: str = Field(description="The UUID of the user", default="")
    created_at: str = Field(description="ISO timestamp when session was created", default="")
    uptime: int = Field(description="Session uptime in nanoseconds", default=0)
    compute_resources: dict = Field(description="Compute resources allocated to session", default={})
    pipeline_ttl: int | None = Field(description="The ttl of the pipeline", default=None)
    session_active: bool = Field(description="Whether the session is active", default=False)
