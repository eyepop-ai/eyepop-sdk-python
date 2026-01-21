"""Integration tests for the worker endpoint.

These tests require a valid EYEPOP_API_KEY environment variable
and make real API calls to the EyePop service.

Run with: pytest tests/integration/ -v
"""

import os

import pytest

from eyepop import EyePopSdk
from eyepop.worker.worker_types import InferenceComponent, Pop

PUBLIC_TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/800px-Cat_November_2010-1a.jpg"
TEST_POP = Pop(components=[
    InferenceComponent(ability="eyepop.person:latest")
])


def requires_api_key():
    """Skip test if EYEPOP_API_KEY is not set."""
    return pytest.mark.skipif(
        not os.getenv("EYEPOP_API_KEY"),
        reason="EYEPOP_API_KEY environment variable not set",
    )


@requires_api_key()
def test_transient_pop_load_from_url():
    """Test loading an image from URL and getting predictions."""
    with EyePopSdk.workerEndpoint(pop_id="transient") as endpoint:
        endpoint.set_pop(TEST_POP)

        job = endpoint.load_from(PUBLIC_TEST_IMAGE_URL)
        result = job.predict()

        assert result is not None
        assert "source_width" in result
        assert "source_height" in result
        assert result["source_width"] > 0
        assert result["source_height"] > 0


@requires_api_key()
@pytest.mark.asyncio
async def test_transient_pop_load_from_url_async():
    """Test async loading an image from URL and getting predictions."""
    async with EyePopSdk.workerEndpoint(pop_id="transient", is_async=True) as endpoint:
        await endpoint.set_pop(TEST_POP)

        job = await endpoint.load_from(PUBLIC_TEST_IMAGE_URL)
        result = await job.predict()

        assert result is not None
        assert "source_width" in result
        assert "source_height" in result
        assert result["source_width"] > 0
        assert result["source_height"] > 0
