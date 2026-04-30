import os

import pytest

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

PUBLIC_TEST_IMAGE_URL = "https://raw.githubusercontent.com/eyepop-ai/eyepop-sdk-python/main/tests/test.jpg"
TEST_POP = Pop(components=[
    InferenceComponent(ability="eyepop.person:latest")
])

# test.jpg is 480x640, model should detect at least 10 persons
EXPECTED_SOURCE_WIDTH = 480
EXPECTED_SOURCE_HEIGHT = 640
MIN_EXPECTED_PERSONS = 10
MIN_CONFIDENCE = 0.5


def requires_api_key():
    return pytest.mark.skipif(
        not os.getenv("EYEPOP_API_KEY"),
        reason="EYEPOP_API_KEY environment variable not set",
    )


def assert_person_detection_result(result: dict):
    assert result["source_width"] == EXPECTED_SOURCE_WIDTH
    assert result["source_height"] == EXPECTED_SOURCE_HEIGHT

    objects = result.get("objects") or []
    assert len(objects) >= MIN_EXPECTED_PERSONS, (
        f"Expected at least {MIN_EXPECTED_PERSONS} persons, got {len(objects)}"
    )

    for obj in objects:
        assert obj["classLabel"] == "person"
        assert obj["confidence"] >= MIN_CONFIDENCE
        assert obj["width"] > 0
        assert obj["height"] > 0
        assert obj["x"] >= 0
        assert obj["y"] >= 0


@requires_api_key()
def test_transient_pop_load_from_url():
    with EyePopSdk.sync_worker(pop_id="transient") as endpoint:
        endpoint.set_pop(TEST_POP)

        job = endpoint.load_from(PUBLIC_TEST_IMAGE_URL)
        result = job.predict()

        assert_person_detection_result(result)


@requires_api_key()
@pytest.mark.asyncio
async def test_transient_pop_load_from_url_async():
    async with EyePopSdk.async_worker(pop_id="transient") as endpoint:
        await endpoint.set_pop(TEST_POP)

        job = await endpoint.load_from(PUBLIC_TEST_IMAGE_URL)
        result = await job.predict()

        assert_person_detection_result(result)


@requires_api_key()
def test_data_endpoint_connect():
    with EyePopSdk.dataEndpoint() as endpoint:
        session_info = endpoint.session()
        assert session_info is not None


@requires_api_key()
@pytest.mark.asyncio
async def test_data_endpoint_connect_async():
    async with EyePopSdk.dataEndpoint(is_async=True) as endpoint:
        session_info = await endpoint.session()
        assert session_info is not None
