import os
from enum import Enum

from pydantic import BaseModel, Field

from eyepop.settings import settings


class ComputeContext(BaseModel):
    """Context for Compute API operations.

    Core context for a compute API session.
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
    m2m_access_token: str = Field(description="The JWT access token from compute API", default="")
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
    UNKNOWN = "unknown"
    PENDING = "pending"
    PIPELINE_CREATING = "pipeline_creating"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ERROR = "error"

    @classmethod
    def _missing_(cls, value):
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

