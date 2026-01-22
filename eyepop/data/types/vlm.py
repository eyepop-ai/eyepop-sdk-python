from datetime import datetime
from typing import Any, Sequence

from pydantic import BaseModel, Field

from eyepop.data.types.enums import EvaluationStatus, VlmAbilityStatus


class InferRuntimeConfig(BaseModel):
    """Runtime configuration for ad-hoc inference [EXPERIMENTAL].

    Common generation parameters are typed explicitly. Additional HuggingFace
    kwargs are accepted via extra="allow" and accessible via model_extra.
    """

    max_new_tokens: int | None = Field(
        default=200, description="Maximum tokens to generate"
    )
    image_size: int | None = Field(
        default=None,
        description="Target size for max dimension. 0 means auto-calculate.",
    )
    fps: float | None = Field(
        default=None,
        description="Target frames per second for sampling. 0.0 means auto-calculate.",
    )
    max_frames: int | None = Field(
        default=None, description="Maximum number of frames to load"
    )
    min_frames: int | None = Field(
        default=None, description="Minimum number of frames required"
    )
    max_aspect_ratio: float | None = Field(
        default=None, description="Maximum allowed aspect ratio"
    )
    context_length: int | None = Field(
        default=None,
        description="Maximum visual tokens to target for auto-calculation",
    )


class TransformInto(BaseModel):
    classes: Sequence[str] | None = Field(
        default=None,
        description="Transform the text output into categorical classes defined by this class label list",
    )


class InferRequest(BaseModel):
    """Client-facing ad-hoc- inference request model [EXPERIMENTAL]."""

    worker_release: str = Field(
        ...,
        description="Worker release name for routing (e.g., qwen3-prod, smol)",
        examples=["smol"],
    )
    text_prompt: str | None = Field(
        default=None,
        description="Prompt for the VLM",
        examples=[
            "How many distinct human beings are in the scene? Answer with an integer number"
        ],
    )
    config: InferRuntimeConfig = Field(
        default_factory=InferRuntimeConfig,
        description="Runtime configuration for inference (max_new_tokens, temperature, etc.)",
        examples=[InferRuntimeConfig(max_new_tokens=10)],
    )
    refresh: bool = Field(
        default=False,
        description="Bypass cache and force re-inference",
        examples=[False],
    )
    transform_into: TransformInto | None = Field(
        default=None,
        description="Optional instructions to transform the raw VLM text output",
    )


class EvaluateFilter(BaseModel):
    partitions: list[str] | None = Field(
        default=None,
        description="Only evaluate assets in these partitions, ignore others",
    )
    ground_truth_classes: list[str] | None = Field(
        default=None,
        description="Only evaluate assets with one of those classes in the ground truth, ignore others",
    )


class EvaluateRequest(BaseModel):
    ability_uuid: str | None = Field(
        default=None,
        description="The uuid of the ability holding the infer config; mutually exclusive to infer",
    )
    infer: InferRequest | None = Field(
        default=None,
        description="VLM inference config; mutually exclusive to ability_uuid",
    )
    dataset_uuid: str = Field(description="The Uuid dataset to evaluate.")
    filter: EvaluateFilter | None = Field(
        default=None,
        description="Only evaluate assets that matches this filter",
    )
    video_chunk_length_ns: int | None = Field(
        default=None,
        description="Video chunk length in nano seconds",
        examples=[2000000000],
    )
    video_chunk_overlap: float | None = Field(
        default=None,
        lt=1.0,
        description="Video chunk overlap ratio, possibly negative to allow gaps,"
        " e.g. -1.0 to have gaps of the same length as the chunk",
        examples=[0.1],
    )


class EvaluateRunInfo(BaseModel):
    """Runtime information for a competed evaluation request."""

    num_images: int = Field(description="Number of images")
    num_ground_truth_images: int = Field(
        description="Number of images with ground truth"
    )
    num_videos: int = Field(description="Number of videos")
    num_ground_truth_videos: int = Field(
        description="Number of videos with ground truth"
    )
    total_tokens: int = Field(description="Total input tokens (visual + text)")
    visual_tokens: int = Field(description="Visual tokens from all frames")
    text_tokens: int = Field(description="Text tokens from prompt")
    output_tokens: int = Field(
        description="Number of tokens generated in the model output"
    )


class EvaluateResponse(BaseModel):
    """Client-facing API response model for evaluate requests."""

    dataset_uuid: str = Field(description="The Uuid for the evaluated dataset.")
    dataset_version: int = Field(
        description="The version number of the evaluated dataset."
    )
    status: EvaluationStatus = Field(description="The final status of the evaluation.")
    status_message: str | None = Field(
        default=None, description="Optional human readable status message."
    )
    source: str = Field(
        description="Source identifier for the persisted  auto annotation."
    )
    metrics: dict[str, Any] | None = Field(default=None, description="Evaluation metrics")
    run_info: EvaluateRunInfo = Field(
        description="Runtime information, e.g. number of processed assets and used resources"
    )


class InferRunInfo(BaseModel):
    """Runtime information about the inference execution.

    Contains details about processing settings, token usage, and media characteristics.
    """

    fps: float | None = Field(
        default=None, description="Frames per second used for video processing"
    )
    image_size: int | None = Field(
        default=None, description="Maximum dimension used for image/frame resizing"
    )
    total_tokens: int | None = Field(
        default=None, description="Total input tokens (visual + text)"
    )
    visual_tokens: int | None = Field(
        default=None, description="Visual tokens from all frames"
    )
    text_tokens: int | None = Field(
        default=None, description="Text tokens from prompt"
    )
    output_tokens: int | None = Field(
        default=None, description="Number of tokens generated in the model output"
    )
    aspect_ratio: float | None = Field(
        default=None, description="Aspect ratio of the processed media (width/height)"
    )


class AbilityAliasEntry(BaseModel):
    alias: str
    tag: str


class VlmAbilityCreate(BaseModel):
    name: str
    description: str
    worker_release: str
    text_prompt: str
    transform_into: TransformInto
    config: InferRuntimeConfig
    is_public: bool


class VlmAbilityUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    worker_release: str | None = None
    text_prompt: str | None = None
    transform_into: TransformInto | None = None
    config: InferRuntimeConfig | None = None
    is_public: bool | None = None


class VlmAbilityResponse(BaseModel):
    uuid: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    account_uuid: str
    status: VlmAbilityStatus
    is_public: bool
    name: str
    description: str
    vlm_ability_group_uuid: str | None = None
    worker_release: str
    text_prompt: str
    transform_into: TransformInto
    config: InferRuntimeConfig
    alias_entries: list[AbilityAliasEntry] | None = None


class VlmAbilityGroupCreate(BaseModel):
    name: str
    description: str
    default_alias_name: str | None = None
    default_dataset_uuid: str | None = None


class VlmAbilityGroupUpdate(BaseModel):
    name: str
    description: str
    default_alias_name: str | None = None
    default_dataset_uuid: str | None = None


class VlmAbilityGroupResponse(BaseModel):
    uuid: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    account_uuid: str
    name: str
    description: str
    default_alias_name: str | None = None
    default_dataset_uuid: str | None = None
