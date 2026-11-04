import json

from eyepop.data.types import InferRequest, InferRuntimeConfig
from eyepop.data.types.dataset import AutoAnnotatePrompt
from eyepop.data.types.enums import AutoAnnotateTask

PROMPTS = [AutoAnnotatePrompt(label="helmet", prompt="a safety helmet")]


def _body(request: InferRequest) -> dict:
    # What the data jobs post as the `infer_request` form field.
    return json.loads(json.dumps(request.model_dump(exclude_none=True)))


def test_task_and_prompts_are_sent() -> None:
    body = _body(
        InferRequest(
            worker_release="gpt-6-astra",
            task=AutoAnnotateTask.object_detection,
            prompts=PROMPTS,
        )
    )
    assert body["task"] == "object_detection"
    assert body["prompts"] == [{"label": "helmet", "prompt": "a safety helmet"}]
    assert "text_prompt" not in body


def test_task_without_limit_sends_no_max_new_tokens() -> None:
    body = _body(InferRequest(task=AutoAnnotateTask.image_classification, prompts=PROMPTS))
    assert "max_new_tokens" not in body["config"]


def test_task_keeps_explicit_limit() -> None:
    config = InferRuntimeConfig(max_new_tokens=4096)
    body = _body(
        InferRequest(task=AutoAnnotateTask.object_detection, prompts=PROMPTS, config=config)
    )
    assert body["config"]["max_new_tokens"] == 4096


def test_task_does_not_change_the_callers_config() -> None:
    config = InferRuntimeConfig(image_size=512)
    InferRequest(task=AutoAnnotateTask.object_detection, prompts=PROMPTS, config=config)
    assert config.max_new_tokens == 200


def test_plain_request_unchanged() -> None:
    body = _body(InferRequest(text_prompt="Describe"))
    assert body["config"]["max_new_tokens"] == 200
    assert "task" not in body and "prompts" not in body


def test_round_trip_from_json() -> None:
    raw = '{"task": "object_detection", "prompts": [{"label": "a", "prompt": "b"}]}'
    request = InferRequest.model_validate_json(raw)
    assert request.task is AutoAnnotateTask.object_detection
    assert request.prompts == [AutoAnnotatePrompt(label="a", prompt="b")]
    assert request.config.max_new_tokens is None
