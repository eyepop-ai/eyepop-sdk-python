from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, Self, Sequence

from openapi_pydantic import Schema
from pydantic import BaseModel, Field, model_validator

from eyepop.data.types.asset import Area
from eyepop.data.types.dataset import AutoAnnotatePrompt
from eyepop.data.types.enums import AutoAnnotateTask, EvaluationStatus, VlmAbilityStatus


class VlmModel(StrEnum):
    """The VLM a user-defined ability runs on."""

    qwen3 = "qwen3"
    """Qwen3 VL instruct, run by EyePop; the default."""
    gpt_6_astra = "gpt_6_astra"
    """GPT-6 Astra, a remote model; images only."""
    gemini_4_argon = "gemini_4_argon"
    """Gemini 4 Argon, a remote model; images only."""


DEFAULT_VLM_MODEL = VlmModel.qwen3
"""The model of an ability that names neither `model` nor `worker_release`."""

VLM_MODEL_WORKER_RELEASES: dict[VlmModel, str] = {
    VlmModel.qwen3: "qwen3_instruct",
    VlmModel.gpt_6_astra: "gpt-6-astra",
    VlmModel.gemini_4_argon: "gemini-4-argon",
}
"""The worker release that runs each model, as the Dataset API stores it."""

_WORKER_RELEASE_DEPRECATED = "Use `model` instead; `worker_release` will be removed in the next major version."


def normalize_worker_release(worker_release: str) -> str:
    """The comparable form of a worker release name: lowercase, `_` replaced by `-`.

    `qwen3_instruct` and `Qwen3-Instruct` both normalize to `qwen3-instruct`.
    """
    return worker_release.lower().replace("_", "-")


_MODELS_BY_WORKER_RELEASE: dict[str, VlmModel] = {
    normalize_worker_release(release): model for model, release in VLM_MODEL_WORKER_RELEASES.items()
}


def model_for_worker_release(worker_release: str | None) -> VlmModel | None:
    """The model a worker release runs: `qwen3` for `None`, `None` for a release that is no model's."""
    if worker_release is None:
        return DEFAULT_VLM_MODEL
    return _MODELS_BY_WORKER_RELEASE.get(normalize_worker_release(worker_release))


def _check_model_and_worker_release(model: VlmModel | None, worker_release: str | None) -> VlmModel | None:
    """The model that `model` and `worker_release` name together, `None` when neither is set.

    Raises `ValueError` when `worker_release` names no model's release or names another model than `model`.
    """
    if worker_release is None:
        return model
    release_model = model_for_worker_release(worker_release)
    if release_model is None:
        raise ValueError(
            f"unknown worker_release {worker_release!r}; worker_release is deprecated, "
            f"set `model` to one of {', '.join(m.value for m in VlmModel)}"
        )
    if model is not None and model != release_model:
        raise ValueError(
            f"worker_release {worker_release!r} runs model {release_model.value}, not {model.value}; "
            "worker_release is deprecated, set only `model`"
        )
    return release_model


def resolve_worker_release(model: VlmModel | None, worker_release: str | None) -> str:
    """The worker release to store for an ability: the canonical release of the model the fields name.

    `qwen3_instruct` when neither is set. Raises `ValueError` like `VlmAbilityCreate` does when
    `worker_release` names no model's release or disagrees with `model`.
    """
    resolved = _check_model_and_worker_release(model, worker_release) or DEFAULT_VLM_MODEL
    return VLM_MODEL_WORKER_RELEASES[resolved]


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


Task = Annotated[
    AutoAnnotateTask | None,
    Field(
        description=(
            "Auto-label task. The worker derives the prompt and output schema from the task "
            "and `prompts`, so `text_prompt`, `transform_into` and `response_schema` must be "
            "unset. Object detection returns `objects`, image classification `classes`, each "
            "labeled with a prompt's label. One image per request."
        ),
        examples=[AutoAnnotateTask.object_detection],
    ),
]
"""An auto-label task, shared by `InferRequest` and the VLM ability models."""

TaskPrompts = Annotated[
    list[AutoAnnotatePrompt] | None,
    Field(
        description="Labels and what each one means, for `task`; required with a task",
        examples=[[AutoAnnotatePrompt(label="helmet", prompt="a safety helmet worn by a person")]],
    ),
]
"""The labeled prompts of a `Task`."""


def _leave_token_limit_to_worker(config: InferRuntimeConfig) -> InferRuntimeConfig:
    """The config a task runs with: without an explicit `max_new_tokens`, the worker's own default.

    The 200-token default would cut off a detection answer. Returns a copy, so a config object
    the caller reuses is not changed.
    """
    if "max_new_tokens" in config.model_fields_set:
        return config
    return config.model_copy(update={"max_new_tokens": None})


def _check_task_mode(
    task: AutoAnnotateTask | None,
    prompts: list[AutoAnnotatePrompt] | None,
    *,
    text_prompt: str | None,
    transform_into: TransformInto | None,
    response_schema: Schema | None,
    partial: bool = False,
) -> None:
    """Raises `ValueError` unless the fields describe one mode: free-form or a task.

    A task needs at least one prompt and derives the prompt and output schema, so `text_prompt`,
    `transform_into` and `response_schema` must stay unset; `prompts` mean nothing without a task.
    A `partial` model (an update) holds only the fields it changes: it may change `prompts`
    without `task` and `task` without `prompts`, but not to an empty list.
    """
    if prompts is not None and not prompts:
        raise ValueError("`prompts` require at least one entry")
    if task is None:
        if prompts is not None and not partial:
            raise ValueError("`prompts` require a `task`")
        return
    if prompts is None and not partial:
        raise ValueError(f"task {task} requires at least one entry in `prompts`")
    conflicting = [
        name
        for name, value in (
            ("text_prompt", text_prompt),
            ("transform_into", transform_into),
            ("response_schema", response_schema),
        )
        if value is not None
    ]
    if conflicting:
        raise ValueError(
            f"task {task} derives the prompt and output schema, so {', '.join(conflicting)} must be unset"
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
    response_schema: Schema | None = Field(
        default=None,
        description=(
            "Schema to constrain the output. When set, guarantees "
            "structured object matching this Open API schema. The parsed "
            "object is returned on each prediction's `details` attribute."
        ),
        examples=[
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {"enum": ["dog", "cat", "other"]},
                    "count": {"type": "integer"},
                    "notes": {"type": "string"},
                },
            }
        ],
    )
    task: Task = None
    prompts: TaskPrompts = None

    @model_validator(mode="after")
    def _task_uses_worker_token_limit(self) -> Self:
        if self.task is not None:
            self.config = _leave_token_limit_to_worker(self.config)
        return self


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
    response_schema: Schema | None = Field(
        default=None,
        description=(
            "Schema to constrain the output. When set, guarantees "
            "structured object matching this Open API schema. The parsed "
            "object is returned on each prediction's `details` attribute."
        ),
        examples=[
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {"enum": ["dog", "cat", "other"]},
                    "count": {"type": "integer"},
                    "notes": {"type": "string"},
                },
            }
        ],
    )
    evaluate: EvaluateConfig = Field(
        description="EvaluateConfig for VLM evaluation.",
    )


class TaskType(StrEnum):
    base = "invalid>"
    classification = "classification"


class TaskMode(StrEnum):
    image = "image"
    video = "video"


class BaseTask(BaseModel):
    task_mode: TaskMode | None = Field(
        default=None,
        description="Defines the processing mode for the task ability when run on videos. Either on individual "
                    "images (frames of the video), or on video chunks of automatically determined length."
    )
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
    """A new VLM ability: free-form with a `text_prompt`, or an auto-label `task` with `prompts`.

    A free-form ability takes the `text_prompt` and optionally `transform_into` and
    `response_schema`. A task ability takes `task` and at least one entry in `prompts`; the worker
    derives the prompt and output schema from them, so `text_prompt`, `transform_into` and
    `response_schema` stay unset.
    """

    name: str = Field(
        description="The human readable name of the ability."
    )
    description: str = Field(
        default="",
        description="Human readable optional description of the ability."
    )
    model: VlmModel | None = Field(
        default=None,
        description="The VLM that runs this ability; `qwen3` when neither `model` nor `worker_release` is set.",
        examples=[VlmModel.qwen3],
    )
    worker_release: str | None = Field(
        default=None,
        description="Deprecated: use `model`. The worker release that runs this ability; must be the release "
                    "of one of the models, and agree with `model` when both are set.",
        deprecated=_WORKER_RELEASE_DEPRECATED,
    )
    text_prompt: str | None = Field(
        default=None,
        description="The full text prompt used for the ability; required unless the ability has a `task`.",
    )
    transform_into: TransformInto | None = Field(
        default_factory=TransformInto,
        description="Optional transform instruction of the text result into structured response, e.g. classes. "
                    "None for a task ability.",
    )
    response_schema: Schema | None = Field(
        default=None,
        description=(
            "Schema to constrain the output. When set, guarantees "
            "structured object matching this Open API schema. The parsed "
            "object is returned on each prediction's `details` attribute."
        ),
        examples=[
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {"enum": ["dog", "cat", "other"]},
                    "count": {"type": "integer"},
                    "notes": {"type": "string"},
                },
            }
        ],
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
    task: Task = None
    prompts: TaskPrompts = None

    @model_validator(mode="after")
    def _one_model(self) -> Self:
        # Read through __dict__: the attribute of a deprecated field warns, and the Dataset API runs this too.
        _check_model_and_worker_release(self.model, self.__dict__.get("worker_release"))
        return self

    @model_validator(mode="after")
    def _one_mode(self) -> Self:
        if self.task is None:
            if self.text_prompt is None:
                raise ValueError("an ability without a `task` requires a `text_prompt`")
        elif "transform_into" not in self.model_fields_set:
            # The empty default is for free-form abilities; a task ability has none.
            self.transform_into = None
        _check_task_mode(
            self.task,
            self.prompts,
            text_prompt=self.text_prompt,
            transform_into=self.transform_into,
            response_schema=self.response_schema,
        )
        if self.task is not None:
            self.config = _leave_token_limit_to_worker(self.config)
        return self


class VlmAbilityUpdate(BaseModel):
    name: str | None = Field(
        default=None,
        description="The human readable name of the ability."
    )
    description: str | None = Field(
        default=None,
        description="Human readable optional description of the ability."
    )
    model: VlmModel | None = Field(
        default=None,
        description="The VLM that runs this ability; unchanged when unset.",
        examples=[VlmModel.qwen3],
    )
    worker_release: str | None = Field(
        default=None,
        description="Deprecated: use `model`. The worker release that runs this ability; must be the release "
                    "of one of the models, and agree with `model` when both are set.",
        deprecated=_WORKER_RELEASE_DEPRECATED,
    )
    text_prompt: str | None = Field(
        default=None,
        description="The full text prompt used for the ability."
    )
    transform_into: TransformInto | None = Field(
        default=None,
        description="Optional transform instruction of the text result into structured response, e.g. classes.",
    )
    response_schema: Schema | None = Field(
        default=None,
        description=(
            "Schema to constrain the output. When set, guarantees "
            "structured object matching this Open API schema. The parsed "
            "object is returned on each prediction's `details` attribute."
        ),
        examples=[
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {"enum": ["dog", "cat", "other"]},
                    "count": {"type": "integer"},
                    "notes": {"type": "string"},
                },
            }
        ],
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
    task: Task = None
    prompts: TaskPrompts = None

    @model_validator(mode="after")
    def _one_model(self) -> Self:
        # Read through __dict__: the attribute of a deprecated field warns, and the Dataset API runs this too.
        _check_model_and_worker_release(self.model, self.__dict__.get("worker_release"))
        return self

    @model_validator(mode="after")
    def _one_mode(self) -> Self:
        _check_task_mode(
            self.task,
            self.prompts,
            text_prompt=self.text_prompt,
            transform_into=self.transform_into,
            response_schema=self.response_schema,
            partial=True,
        )
        if self.task is not None and self.config is not None:
            self.config = _leave_token_limit_to_worker(self.config)
        return self


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
    model: VlmModel | None = Field(
        default=None,
        description="The VLM that runs this ability; `None` for an ability whose worker release is no model's.",
    )
    worker_release: str | None = Field(
        default=None,
        description="Deprecated: use `model`. The worker release that runs this ability.",
        deprecated=_WORKER_RELEASE_DEPRECATED,
    )
    text_prompt: str | None = Field(
        default=None,
        description="The full text prompt used for the ability."
    )
    transform_into: TransformInto | None = Field(
        default=None,
        description="Optional transform instruction of the text result into structured response, e.g. classes.",
    )
    response_schema: Schema | None = Field(
        default=None,
        description=(
            "Schema to constrain the output. When set, guarantees "
            "structured object matching this Open API schema. The parsed "
            "object is returned on each prediction's `details` attribute."
        ),
        examples=[
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["label"],
                "properties": {
                    "label": {"enum": ["dog", "cat", "other"]},
                    "count": {"type": "integer"},
                    "notes": {"type": "string"},
                },
            }
        ],
    )
    config: InferRuntimeConfig | None = Field(
        default=None,
        description="Optional inference configuration for VLM inference of this ability.",
    )
    task: Task = None
    prompts: TaskPrompts = None
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
        description="Optionally, the auto prompt config this ability was created or refined  with.",
        default=None,
        deprecated="Use auto_task instead",
    )
    auto_task: AutoTask | None = Field(
        description="Optionally, the auto task config this ability was created or refined  with."
                    "Replaces the deprecated auto_prompt.",
        default=None,
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_model(cls, data: Any) -> Any:
        # A Dataset API from before `model` sends only the worker release.
        if isinstance(data, dict) and "model" not in data:
            return {**data, "model": model_for_worker_release(data.get("worker_release"))}
        return data


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
