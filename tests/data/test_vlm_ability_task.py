import json
from typing import Any

import pytest
from openapi_pydantic import Schema
from pydantic import ValidationError

from eyepop.data.data_endpoint import _vlm_ability_body
from eyepop.data.types import (
    InferRuntimeConfig,
    TransformInto,
    VlmAbilityCreate,
    VlmAbilityResponse,
    VlmAbilityUpdate,
    VlmModel,
)
from eyepop.data.types.dataset import AutoAnnotatePrompt
from eyepop.data.types.enums import AutoAnnotateTask

PROMPTS = [AutoAnnotatePrompt(label="helmet", prompt="a safety helmet")]


def _create(**fields: Any) -> VlmAbilityCreate:
    return VlmAbilityCreate(name="helmets", model=VlmModel.gpt_6_astra, **fields)


def _create_body(create: VlmAbilityCreate) -> dict[str, Any]:
    # What DataEndpoint.create_vlm_ability posts.
    return json.loads(_vlm_ability_body(create, exclude_none=True))


def _update_body(update: VlmAbilityUpdate) -> dict[str, Any]:
    # What DataEndpoint.update_vlm_ability patches.
    return json.loads(_vlm_ability_body(update, exclude_none=False))


def test_a_free_form_create_is_unchanged() -> None:
    create = _create(text_prompt="Is there a helmet?")

    assert create.transform_into == TransformInto()
    assert create.config.max_new_tokens == 200
    assert _create_body(create) == {
        "name": "helmets",
        "model": "gpt_6_astra",
        "worker_release": "gpt-6-astra",
        "text_prompt": "Is there a helmet?",
    }


def test_a_free_form_create_keeps_transform_into_and_response_schema() -> None:
    create = _create(
        text_prompt="Is there a helmet?",
        transform_into=TransformInto(classes=["yes", "no"]),
        response_schema=Schema(type="object"),
    )

    body = _create_body(create)
    assert body["transform_into"] == {"classes": ["yes", "no"]}
    assert body["response_schema"]["type"] == "object"
    assert "task" not in body and "prompts" not in body


@pytest.mark.parametrize("task", list(AutoAnnotateTask))
def test_a_task_create_sends_task_and_prompts_only(task: AutoAnnotateTask) -> None:
    create = _create(task=task, prompts=PROMPTS)

    assert create.text_prompt is None
    assert create.transform_into is None
    body = _create_body(create)
    assert body["task"] == task.value
    assert body["prompts"] == [{"label": "helmet", "prompt": "a safety helmet"}]
    assert "text_prompt" not in body
    assert "transform_into" not in body
    assert "response_schema" not in body


def test_a_task_create_leaves_the_token_limit_to_the_worker() -> None:
    assert _create(task=AutoAnnotateTask.object_detection, prompts=PROMPTS).config.max_new_tokens is None

    config = InferRuntimeConfig(max_new_tokens=4096)
    create = _create(task=AutoAnnotateTask.object_detection, prompts=PROMPTS, config=config)
    assert create.config.max_new_tokens == 4096


def test_a_task_create_does_not_change_the_callers_config() -> None:
    config = InferRuntimeConfig(image_size=512)
    _create(task=AutoAnnotateTask.object_detection, prompts=PROMPTS, config=config)
    assert config.max_new_tokens == 200


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({}, "requires a `text_prompt`"),
        ({"prompts": PROMPTS}, "requires a `text_prompt`"),
        ({"text_prompt": "x", "prompts": PROMPTS}, "`prompts` require a `task`"),
        ({"task": AutoAnnotateTask.object_detection}, "requires at least one entry in `prompts`"),
        ({"task": AutoAnnotateTask.object_detection, "prompts": []}, "require at least one entry"),
        ({"task": AutoAnnotateTask.object_detection, "prompts": PROMPTS, "text_prompt": "x"}, "text_prompt must be unset"),
        (
            {"task": AutoAnnotateTask.object_detection, "prompts": PROMPTS, "transform_into": TransformInto()},
            "transform_into must be unset",
        ),
        (
            {"task": AutoAnnotateTask.object_detection, "prompts": PROMPTS, "response_schema": Schema(type="object")},
            "response_schema must be unset",
        ),
    ],
)
def test_a_create_is_free_form_or_a_task(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _create(**fields)


def test_a_task_create_round_trips_through_json() -> None:
    raw = (
        '{"name": "helmets", "model": "gpt_6_astra", "task": "image_classification",'
        ' "prompts": [{"label": "a", "prompt": "b"}]}'
    )
    create = VlmAbilityCreate.model_validate_json(raw)

    assert create.task is AutoAnnotateTask.image_classification
    assert create.prompts == [AutoAnnotatePrompt(label="a", prompt="b")]
    assert create.transform_into is None
    assert VlmAbilityCreate.model_validate_json(create.model_dump_json()) == create


def test_an_update_sends_task_and_prompts() -> None:
    update = VlmAbilityUpdate(task=AutoAnnotateTask.object_detection, prompts=PROMPTS)

    assert _update_body(update) == {
        "task": "object_detection",
        "prompts": [{"label": "helmet", "prompt": "a safety helmet"}],
    }


def test_an_update_may_change_prompts_alone() -> None:
    assert _update_body(VlmAbilityUpdate(prompts=PROMPTS)) == {
        "prompts": [{"label": "helmet", "prompt": "a safety helmet"}],
    }


def test_a_free_form_update_is_unchanged() -> None:
    assert _update_body(VlmAbilityUpdate(text_prompt="x")) == {"text_prompt": "x"}


@pytest.mark.parametrize(
    "fields",
    [
        {"prompts": []},
        {"task": AutoAnnotateTask.object_detection, "text_prompt": "x"},
        {"task": AutoAnnotateTask.object_detection, "transform_into": TransformInto()},
        {"task": AutoAnnotateTask.object_detection, "response_schema": Schema(type="object")},
    ],
)
def test_an_update_rejects_a_conflicting_patch(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VlmAbilityUpdate(**fields)


def test_a_response_reads_task_and_prompts() -> None:
    response = VlmAbilityResponse.model_validate(
        {
            "uuid": "u",
            "account_uuid": "a",
            "status": "draft",
            "name": "helmets",
            "task": "object_detection",
            "prompts": [{"label": "helmet", "prompt": "a safety helmet"}],
        }
    )

    assert response.task is AutoAnnotateTask.object_detection
    assert response.prompts == PROMPTS
    assert response.text_prompt is None


def test_a_free_form_response_has_no_task() -> None:
    response = VlmAbilityResponse.model_validate(
        {"uuid": "u", "account_uuid": "a", "status": "draft", "name": "n", "text_prompt": "x"}
    )

    assert response.task is None and response.prompts is None
