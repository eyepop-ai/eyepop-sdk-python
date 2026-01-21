import os

import pytest

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

PUBLIC_TEST_IMAGE_URL = "https://raw.githubusercontent.com/eyepop-ai/eyepop-sdk-python/main/tests/test.jpg"
TEST_POP = Pop(components=[
    InferenceComponent(ability="eyepop.person:latest")
])


def requires_api_key():
    return pytest.mark.skipif(
        not os.getenv("EYEPOP_API_KEY"),
        reason="EYEPOP_API_KEY environment variable not set",
    )


@requires_api_key()
def test_transient_pop_load_from_url():
    with EyePopSdk.workerEndpoint(pop_id="transient") as endpoint:
        endpoint.set_pop(TEST_POP)

        job = endpoint.load_from(PUBLIC_TEST_IMAGE_URL)
        result = job.predict()

        print("\n=== Inference Result (sync) ===")
        print(f"Source: {result.get('source_width')}x{result.get('source_height')}")
        print(f"Objects detected: {len(result.get('objects', []))}")
        for obj in result.get("objects", []):
            print(f"  - {obj.get('classLabel')} (confidence: {obj.get('confidence', 0):.2f})")

        assert result is not None
        assert "source_width" in result
        assert "source_height" in result
        assert result["source_width"] > 0
        assert result["source_height"] > 0


@requires_api_key()
@pytest.mark.asyncio
async def test_transient_pop_load_from_url_async():
    async with EyePopSdk.workerEndpoint(pop_id="transient", is_async=True) as endpoint:
        await endpoint.set_pop(TEST_POP)

        job = await endpoint.load_from(PUBLIC_TEST_IMAGE_URL)
        result = await job.predict()

        print("\n=== Inference Result (async) ===")
        print(f"Source: {result.get('source_width')}x{result.get('source_height')}")
        print(f"Objects detected: {len(result.get('objects', []))}")
        for obj in result.get("objects", []):
            print(f"  - {obj.get('classLabel')} (confidence: {obj.get('confidence', 0):.2f})")

        assert result is not None
        assert "source_width" in result
        assert "source_height" in result
        assert result["source_width"] > 0
        assert result["source_height"] > 0


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
