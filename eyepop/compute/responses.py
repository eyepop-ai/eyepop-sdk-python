from pydantic import BaseModel, Field

from eyepop.compute.context import PipelineStatus


class ComputeApiPipelineStatus(BaseModel):
    status: PipelineStatus = Field(description="The status of the pipeline")
    reason: str = Field(description="The reason for the status")


class ComputeApiSessionRequest(BaseModel):
    account_uuid: str = Field(description="Required account uuid to create a session")


class ComputeApiSessionResponse(BaseModel):
    session_uuid: str = Field(description="The related session uuid for this session")
    session_endpoint: str = Field(description="The related session url for this session")
    access_token: str = Field(description="The JWT access token for session authentication")
    access_token_expires_at: str = Field(
        description="ISO timestamp when access token expires", default=""
    )
    access_token_expires_in: int = Field(
        description="Seconds until access token expires", default=0
    )
    access_token_valid_until: int | None = Field(
        description="The timestamp of the access token valid until", default=None
    )
    pipelines: list[dict] = Field(description="List of pipelines in the session", default=[])
    pipeline_uuid: str = Field(description="The related pipeline uuid for this session", default="")
    pipeline_version: str = Field(
        description="The related pipeline version for this session", default=""
    )
    session_status: PipelineStatus = Field(
        description="The status of the session", default=PipelineStatus.PENDING
    )
    session_message: str = Field(description="The message of the session", default="")
    session_name: str = Field(description="The name of the session", default="")
    user_uuid: str = Field(description="The UUID of the user", default="")
    created_at: str = Field(description="ISO timestamp when session was created", default="")
    uptime: int = Field(description="Session uptime in nanoseconds", default=0)
    compute_resources: dict = Field(
        description="Compute resources allocated to session", default={}
    )
    pipeline_ttl: int | None = Field(description="The ttl of the pipeline", default=None)
    session_active: bool = Field(description="Whether the session is active", default=False)
