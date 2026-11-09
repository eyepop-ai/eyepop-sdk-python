"""`model` names the VLM of a user-defined ability; `worker_release` is deprecated (AIW-169)."""

import json
import warnings
from typing import Any

import pytest
from aioresponses import aioresponses
from pydantic import ValidationError

from eyepop import EyePopSdk
from eyepop.data.types import (
    DEFAULT_VLM_MODEL,
    VLM_MODEL_WORKER_RELEASES,
    VlmAbilityCreate,
    VlmAbilityResponse,
    VlmAbilityUpdate,
    VlmModel,
    model_for_worker_release,
    normalize_worker_release,
    resolve_worker_release,
)
from tests.data.base_endpoint_test import BaseEndpointTest

ABILITY_UUID = "test_ability_uuid"


def _create(**fields: Any) -> VlmAbilityCreate:
    return VlmAbilityCreate(name="n", text_prompt="x", **fields)


def _response(**fields: Any) -> VlmAbilityResponse:
    return VlmAbilityResponse.model_validate(
        {"uuid": ABILITY_UUID, "account_uuid": "a", "status": "draft", "name": "n", "text_prompt": "x", **fields}
    )


# Helpers


def test_each_model_maps_to_its_worker_release() -> None:
    assert VLM_MODEL_WORKER_RELEASES == {
        VlmModel.qwen3: "qwen3_instruct",
        VlmModel.gpt_6_astra: "gpt-6-astra",
        VlmModel.gemini_4_argon: "gemini-4-argon",
    }
    assert DEFAULT_VLM_MODEL is VlmModel.qwen3


def test_normalize_worker_release_lowercases_and_uses_dashes() -> None:
    assert normalize_worker_release("Qwen3_Instruct") == "qwen3-instruct"
    assert normalize_worker_release("gpt-6-astra") == "gpt-6-astra"


@pytest.mark.parametrize(
    ("worker_release", "model"),
    [
        (None, VlmModel.qwen3),
        ("qwen3_instruct", VlmModel.qwen3),
        ("qwen3-instruct", VlmModel.qwen3),
        ("QWEN3-INSTRUCT", VlmModel.qwen3),
        ("gpt-6-astra", VlmModel.gpt_6_astra),
        ("gpt_6_astra", VlmModel.gpt_6_astra),
        ("gemini-4-argon", VlmModel.gemini_4_argon),
        ("gemini_4_argon", VlmModel.gemini_4_argon),
        ("smol22b-instruct", None),
        ("qwen3", None),
    ],
)
def test_model_for_worker_release(worker_release: str | None, model: VlmModel | None) -> None:
    assert model_for_worker_release(worker_release) is model


@pytest.mark.parametrize(
    ("model", "worker_release", "resolved"),
    [
        (None, None, "qwen3_instruct"),
        (VlmModel.qwen3, None, "qwen3_instruct"),
        (VlmModel.gpt_6_astra, None, "gpt-6-astra"),
        (VlmModel.gemini_4_argon, None, "gemini-4-argon"),
        (None, "qwen3-instruct", "qwen3_instruct"),
        (None, "GPT_6_ASTRA", "gpt-6-astra"),
        (VlmModel.gemini_4_argon, "gemini_4_argon", "gemini-4-argon"),
    ],
)
def test_resolve_worker_release_is_the_canonical_release(
    model: VlmModel | None, worker_release: str | None, resolved: str
) -> None:
    assert resolve_worker_release(model, worker_release) == resolved


def test_resolve_worker_release_refuses_an_unknown_or_disagreeing_release() -> None:
    with pytest.raises(ValueError, match="unknown worker_release 'smol22b-instruct'"):
        resolve_worker_release(None, "smol22b-instruct")
    with pytest.raises(ValueError, match="runs model qwen3, not gpt_6_astra"):
        resolve_worker_release(VlmModel.gpt_6_astra, "qwen3_instruct")


# Create and update models


def test_a_create_needs_neither_model_nor_worker_release() -> None:
    create = _create()
    assert create.model is None
    assert resolve_worker_release(create.model, create.__dict__["worker_release"]) == "qwen3_instruct"


@pytest.mark.parametrize("model", list(VlmModel))
def test_a_create_takes_a_model(model: VlmModel) -> None:
    assert _create(model=model).model is model


def test_a_create_takes_the_model_by_value() -> None:
    assert VlmAbilityCreate.model_validate({"name": "n", "text_prompt": "x", "model": "gemini_4_argon"}).model is (
        VlmModel.gemini_4_argon
    )
    with pytest.raises(ValidationError):
        VlmAbilityCreate.model_validate({"name": "n", "text_prompt": "x", "model": "gemini-4-argon"})


@pytest.mark.parametrize("worker_release", ["qwen3_instruct", "qwen3-instruct", "gpt-6-astra", "Gemini_4_Argon"])
def test_a_create_still_takes_a_known_worker_release(worker_release: str) -> None:
    create = VlmAbilityCreate.model_validate({"name": "n", "text_prompt": "x", "worker_release": worker_release})
    assert create.__dict__["worker_release"] == worker_release


@pytest.mark.parametrize(
    ("model", "worker_release"),
    [
        (VlmModel.qwen3, "qwen3-instruct"),
        (VlmModel.qwen3, "qwen3_instruct"),
        (VlmModel.gpt_6_astra, "gpt_6_astra"),
        (VlmModel.gemini_4_argon, "gemini-4-argon"),
    ],
)
def test_model_and_worker_release_may_agree(model: VlmModel, worker_release: str) -> None:
    _create(model=model, worker_release=worker_release)
    VlmAbilityUpdate(model=model, worker_release=worker_release)


@pytest.mark.parametrize("make", [_create, VlmAbilityUpdate])
def test_model_and_worker_release_must_agree(make: Any) -> None:
    with pytest.raises(ValidationError, match="worker_release 'gpt-6-astra' runs model gpt_6_astra, not qwen3"):
        make(model=VlmModel.qwen3, worker_release="gpt-6-astra")


@pytest.mark.parametrize("make", [_create, VlmAbilityUpdate])
def test_an_unknown_worker_release_is_refused(make: Any) -> None:
    with pytest.raises(ValidationError, match="unknown worker_release 'smol22b-instruct'.*set `model`"):
        make(worker_release="smol22b-instruct")


def test_an_update_that_leaves_the_model_alone_validates() -> None:
    update = VlmAbilityUpdate(name="renamed")
    assert update.model is None
    assert "model" not in update.model_fields_set and "worker_release" not in update.model_fields_set


def test_validating_a_worker_release_does_not_warn() -> None:
    # The Dataset API validates the same models; the deprecation warning is the client's.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _create(worker_release="gpt-6-astra")
        VlmAbilityUpdate(worker_release="gpt-6-astra")
        _response(worker_release="gpt-6-astra")


def test_worker_release_is_deprecated_in_the_json_schema() -> None:
    for model in (VlmAbilityCreate, VlmAbilityUpdate, VlmAbilityResponse):
        assert model.model_json_schema()["properties"]["worker_release"]["deprecated"] is True


# Response


@pytest.mark.parametrize(
    ("fields", "model"),
    [
        ({"model": "gpt_6_astra", "worker_release": "gpt-6-astra"}, VlmModel.gpt_6_astra),
        ({"model": None, "worker_release": "smol22b-instruct"}, None),
        # A Dataset API from before `model` sends only the release.
        ({"worker_release": "gemini-4-argon"}, VlmModel.gemini_4_argon),
        ({"worker_release": "smol22b-instruct"}, None),
        ({}, VlmModel.qwen3),
    ],
)
def test_a_response_reports_the_model(fields: dict[str, Any], model: VlmModel | None) -> None:
    assert _response(**fields).model is model


# Client


class TestVlmAbilityEndpoint(BaseEndpointTest):
    def setup_mock(self, mock: aioresponses) -> None:
        self.setup_base_mock(mock)
        mock.post(
            f"{self.test_eyepop_url}/authentication/token",
            status=200,
            body=json.dumps({"expires_in": 1000 * 1000, "token_type": "Bearer", "access_token": self.test_access_token}),
        )
        body = _response(model="gpt_6_astra", worker_release="gpt-6-astra").model_dump_json()
        mock.post(
            f"{self.test_data_url}/vlm_abilities?account_uuid={self.test_eyepop_account_id}", status=200, body=body
        )
        mock.patch(f"{self.test_data_url}/vlm_abilities/{ABILITY_UUID}", status=200, body=body)

    def sent(self, mock: aioresponses, method: str, path: str) -> dict[str, Any]:
        calls = [call for key, calls in mock.requests.items() if key[0] == method and key[1].path == path for call in calls]
        self.assertEqual(len(calls), 1)
        return json.loads(calls[0].kwargs["data"])

    def endpoint(self):
        return EyePopSdk.dataEndpoint(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            account_id=self.test_eyepop_account_id,
        )

    @aioresponses()
    def test_a_create_with_a_model_also_sends_its_worker_release(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        with self.endpoint() as endpoint, warnings.catch_warnings():
            warnings.filterwarnings("error", message="worker_release", category=DeprecationWarning)
            created = endpoint.create_vlm_ability(_create(model=VlmModel.gpt_6_astra))

        self.assertIs(created.model, VlmModel.gpt_6_astra)
        self.assertEqual(
            self.sent(mock, "POST", "/vlm_abilities"),
            {"name": "n", "text_prompt": "x", "model": "gpt_6_astra", "worker_release": "gpt-6-astra"},
        )

    @aioresponses()
    def test_a_create_without_a_model_sends_neither(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        with self.endpoint() as endpoint:
            endpoint.create_vlm_ability(_create())

        self.assertEqual(self.sent(mock, "POST", "/vlm_abilities"), {"name": "n", "text_prompt": "x"})

    @aioresponses()
    def test_a_create_with_a_worker_release_warns_and_sends_it(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        with self.endpoint() as endpoint:
            with self.assertWarnsRegex(DeprecationWarning, "worker_release is deprecated") as caught:
                endpoint.create_vlm_ability(_create(worker_release="qwen3-instruct"))

        self.assertEqual(caught.filename, __file__)
        self.assertEqual(
            self.sent(mock, "POST", "/vlm_abilities"),
            {"name": "n", "text_prompt": "x", "worker_release": "qwen3-instruct"},
        )

    @aioresponses()
    def test_an_update_with_a_model_also_sends_its_worker_release(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        with self.endpoint() as endpoint, warnings.catch_warnings():
            warnings.filterwarnings("error", message="worker_release", category=DeprecationWarning)
            endpoint.update_vlm_ability(ABILITY_UUID, VlmAbilityUpdate(model=VlmModel.gemini_4_argon))

        self.assertEqual(
            self.sent(mock, "PATCH", f"/vlm_abilities/{ABILITY_UUID}"),
            {"model": "gemini_4_argon", "worker_release": "gemini-4-argon"},
        )

    @aioresponses()
    def test_an_update_without_a_model_leaves_it_alone(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        with self.endpoint() as endpoint:
            endpoint.update_vlm_ability(ABILITY_UUID, VlmAbilityUpdate(name="renamed"))

        self.assertEqual(self.sent(mock, "PATCH", f"/vlm_abilities/{ABILITY_UUID}"), {"name": "renamed"})

    @aioresponses()
    def test_an_update_with_a_worker_release_warns(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        with self.endpoint() as endpoint:
            with self.assertWarnsRegex(DeprecationWarning, "worker_release is deprecated") as caught:
                endpoint.update_vlm_ability(ABILITY_UUID, VlmAbilityUpdate(worker_release="gpt_6_astra"))

        self.assertEqual(caught.filename, __file__)
        self.assertEqual(self.sent(mock, "PATCH", f"/vlm_abilities/{ABILITY_UUID}"), {"worker_release": "gpt_6_astra"})

    @aioresponses()
    async def test_the_async_client_warns_at_the_caller(self, mock: aioresponses) -> None:
        self.setup_mock(mock)
        async with EyePopSdk.dataEndpoint(
            eyepop_url=self.test_eyepop_url,
            secret_key=self.test_eyepop_secret_key,
            account_id=self.test_eyepop_account_id,
            is_async=True,
        ) as endpoint:
            with self.assertWarnsRegex(DeprecationWarning, "worker_release is deprecated") as caught:
                await endpoint.create_vlm_ability(_create(worker_release="gpt-6-astra"))

        self.assertEqual(caught.filename, __file__)
