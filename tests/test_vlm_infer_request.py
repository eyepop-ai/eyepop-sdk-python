import json

import pytest

from eyepop.data.data_jobs import InferJob
from eyepop.data.data_types import Prediction
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


class _Response:
    status = 200

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


class _Session:
    """Answers every infer request 200 with `body`."""

    def __init__(self, body: dict) -> None:
        self._body = body

    async def request_with_retry(self, **_: object) -> _Response:
        return _Response(self._body)


async def _infer_job(body: dict) -> "InferJob":
    job = InferJob("https://images.example/a.jpg", InferRequest(worker_release="qwen3-instruct"), _Session(body))  # type: ignore[arg-type]
    await job.execute()
    return job


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"milli_compute_units": 331, "compute_units": 330}, 331),
        ({"compute_units": 330}, 330),
        ({"milli_compute_units": None, "compute_units": 330}, 330),
        ({}, None),
    ],
)
async def test_an_infer_job_has_the_milli_compute_units_vlm_server_billed(usage: dict, expected: int | None) -> None:
    job = await _infer_job({"predictions": [{"source_width": 1, "source_height": 1}], **usage})

    assert job.milli_compute_units == expected


def test_a_prediction_keeps_its_compute_units() -> None:
    prediction = Prediction.model_validate(
        {"source_width": 1, "source_height": 1, "compute_units": 0.3, "milli_compute_units": 300}
    )

    assert (prediction.compute_units, prediction.milli_compute_units) == (0.3, 300)
