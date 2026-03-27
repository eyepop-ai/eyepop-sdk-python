from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, Field

from eyepop.data.types.asset import Area
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
    roi: Area | None = Field(
        default=None,
        description="Region of interest for cropping",
    )


class TransformInto(BaseModel):
    classes: Sequence[str] | None = Field(
        default=None,
        description="Transform the text output into categorical classes defined by this class label list",
    )


class InferRequest(BaseModel):
    """Client-facing ad-hoc- inference request model [EXPERIMENTAL]."""

    worker_release: str = Field(
        default="qwen3-instruct",
        description="Worker release name for routing",
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


class EvaluateConfig(BaseModel):
    source: str | None = Field(
        default=None,
        description="Optional source identifier to be used for the auto annotations",
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
        lt=0.5,
        description="Video chunk overlap ratio, possibly negative to allow gaps,"
        " e.g. -1.0 to have gaps of the same length as the chunk",
        examples=[0.1],
    )


class EvaluateRequest(EvaluateConfig):
    ability_uuid: str | None = Field(
        default=None,
        description="The uuid of the ability holding the infer config; mutually exclusive to infer",
    )
    infer: InferRequest | None = Field(
        default=None,
        description="VLM inference config; mutually exclusive to ability_uuid",
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


class AutoPromptConfig(BaseModel):
    num_samples: int = Field(
        default=5,
        description="For initial auto prompt creation this is the umber of chunks to sample per class. "
                    "For auto prompt refine this is the number of failure samples to use per mismatch type (FP/FN).",
    )
    task_description: str | None = Field(
        default=None,
        description="Optional task description to append to the LLM prompt for customizing prompt creation",
    )
    infer: InferRequest = Field(
        description="InferConfig for VLM inference. The transform_into.classes field "
        "specifies the possible labels for classification.",
    )
    evaluate: EvaluateConfig = Field(
        description="EvaluateConfig for VLM evaluation.",
    )


class TaskType(StrEnum):
    base = "invalid>"
    classification = "classification"


class BaseTask(BaseModel):
    task_description: str | None = Field(
        default=None,
        description="Optional task description to append to the LLM prompt for customizing prompt creation",
    )


class ClassificationTask(BaseTask):
    type: Literal[TaskType.classification] = TaskType.classification
    classes: Sequence[str] = Field(
        description="List of class names."
    )
    dataset_uuid: str = Field(description="The Uuid dataset to use as input for this auto task.")
    num_samples: int = Field(
        default=5,
        description="For initial auto prompt creation this is the umber of chunks to sample per class. "
                    "For auto prompt refine this is the number of failure samples to use per mismatch type (FP/FN).",
    )


AutoTask = Annotated[ClassificationTask, Field(discriminator="type")]


class AbilityAliasEntry(BaseModel):
    alias: str
    tag: str


class VlmAbilityCreate(BaseModel):
    name: str = Field(
        description="The human readable name of the ability."
    )
    description: str = Field(
        default="",
        description="Human readable optional description of the ability."
    )
    worker_release: str = Field(
        description="The identifier of the worker release used to execute this ability."
    )
    text_prompt: str = Field(
        description="The full text prompt used for the ability."
    )
    transform_into: TransformInto = Field(
        default_factory=TransformInto,
        description="Optional transform instruction of the text result into structured response, e.g. classes.",
    )
    config: InferRuntimeConfig = Field(
        default_factory=InferRuntimeConfig,
        description="Optional inference configuration for VLM inference of this ability.",
    )
    is_public: bool = Field(
        default=False,
        description="Whether or not the ability is publicly accessible.",
    )
    video_chunk_length_ns: int | None = Field(
        default=None,
        description="Video chunk length in nano seconds",
        examples=[2000000000],
    )
    video_chunk_overlap: float | None = Field(
        default=None,
        lt=0.5,
        description="Video chunk overlap ratio, possibly negative to allow gaps,"
        " e.g. -1.0 to have gaps of the same length as the chunk",
        examples=[0.1],
    )


class VlmAbilityUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        description="The human readable name of the ability."
    )
    description: str | None = Field(
        default=None,
        description="Human readable optional description of the ability."
    )
    worker_release: str | None = Field(
        default=None,
        description="The identifier of the worker release used to execute this ability."
    )
    text_prompt: str | None = Field(
        default=None,
        description="The full text prompt used for the ability."
    )
    transform_into: TransformInto | None = Field(
        default=None,
        description="Optional transform instruction of the text result into structured response, e.g. classes.",
    )
    config: InferRuntimeConfig | None = Field(
        default=None,
        description="Optional inference configuration for VLM inference of this ability.",
    )
    is_public: bool | None = Field(
        default=None,
        description="Whether or not the ability is publicly accessible.",
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


class VlmAbilityResponse(BaseModel):
    uuid: str = Field(
        description="The uuid of the ability."
    )
    created_at: datetime | None = Field(
        default=None,
        description="The datetime when the ability was created."
    )
    updated_at: datetime | None = Field(
        default=None,
        description="The datetime when the ability was last updated."
    )
    account_uuid: str = Field(
        description="The uuid of the account associated with the ability."
    )
    status: VlmAbilityStatus = Field(
        description="The status of the ability."
    )
    vlm_ability_group_uuid: str | None = Field(
        default=None,
        description="The uuid of the VLM ability group associated with the ability."
    )
    name: str = Field(
        description="The human readable name of the ability."
    )
    description: str = Field(
        default="",
        description="Human readable optional description of the ability."
    )
    worker_release: str | None = Field(
        default=None,
        description="The identifier of the worker release used to execute this ability."
    )
    text_prompt: str | None = Field(
        default=None,
        description="The full text prompt used for the ability."
    )
    transform_into: TransformInto | None = Field(
        default=None,
        description="Optional transform instruction of the text result into structured response, e.g. classes.",
    )
    config: InferRuntimeConfig | None = Field(
        default=None,
        description="Optional inference configuration for VLM inference of this ability.",
    )
    is_public: bool = Field(
        default=False,
        description="Whether or not the ability is publicly accessible.",
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
    alias_entries: list[AbilityAliasEntry] = Field(
        default=[],
        description="The list of alias entries assigned to this ability."
    )
    auto_prompt: AutoPromptConfig | None = Field(
        description="Optionally, the auto promp config this ability was created or refined  with.",
        default=None,
    )


class VlmAbilityGroupCreate(BaseModel):
    name: str = Field(
        description="Human readable name of the ability group",
    )
    description: str = Field(
        default="",
        description="Optional human readable description of the ability group",
    )

    auto_prompt: AutoPromptConfig | None = Field(
        description="Optionally create an ability in this group via auto prompt agent. "
                    "The created ability will stay in 'in_progress' state until the agent completes.",
        default=None,
        deprecated="Use auto_task instead"
    )

    auto_task: AutoTask | None = Field(
        description="Optionally create an ability in this group via auto task agent. "
                    "The created ability will stay in 'in_progress' state until the agent completes."
                    "Replaces the deprecated auto_prompt.",
        default=None,
    )

    default_alias_name: str | None = Field(
        default=None,
        description="Optionally use this name as default alias name for all abilities in this group",
    )
    default_dataset_uuid: str | None = Field(
        default=None,
        description="Optionally use this dataset UUID as evaluation target for all abilities in this group",
    )


class VlmAbilityGroupUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        description="Human readable name of the ability group",
    )
    description: str | None = Field(
        default=None,
        description="Optional human readable description of the ability group",
    )
    default_alias_name: str | None = Field(
        default=None,
        description="Optionally use this name as default alias name for all abilities in this group",
    )
    default_dataset_uuid: str | None = Field(
        default=None,
        description="Optionally use this dataset UUID as evaluation target for all abilities in this group",
    )


class VlmAbilityGroupResponse(BaseModel):
    uuid: str = Field(
        description="The uuid of the ability group."
    )
    created_at: datetime | None = Field(
        default=None,
        description="The datetime when the ability group was created.",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="The datetime when the ability group was last updated.",
    )
    account_uuid: str = Field(
        description="The uuid of the account associated with the ability group."
    )
    name: str = Field(
        description="Human readable name of the ability group."
    )
    description: str = Field(
        default="",
        description="Optional human readable description of the ability group",
    )
    default_alias_name: str | None = Field(
        default=None,
        description="Name as default alias name for all abilities in this group",
    )
    default_dataset_uuid: str | None = Field(
        default=None,
        description="Dataset UUID as default evaluation target for all abilities in this group",
    )
