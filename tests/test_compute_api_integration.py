import os
from unittest.mock import patch

import aiohttp
import pytest

from eyepop.compute import ComputeContext
from eyepop.compute.api import fetch_new_compute_session, fetch_session_endpoint
from eyepop.exceptions import ComputeSessionException

MOCK_SESSION_RESPONSE = {
    "session_uuid": "session-456",
    "session_endpoint": "https://integration-pipeline.example.com",
    "access_token": "jwt-token-123",
    "pipelines": [{"pipeline_id": "pipeline-123"}],
    "pipeline_uuid": "pipeline-123",
    "pipeline_version": "1.0.0",
    "session_status": "running",
    "session_message": "Session created successfully",
    "pipeline_ttl": 3600,
    "session_active": True
}


@pytest.fixture
def clean_environment():
    original_url = os.environ.get("EYEPOP_URL")
    original_key = os.environ.get("EYEPOP_API_KEY")

    if "EYEPOP_URL" in os.environ:
        del os.environ["EYEPOP_URL"]
    if "EYEPOP_API_KEY" in os.environ:
        del os.environ["EYEPOP_API_KEY"]

    yield

    if "EYEPOP_URL" in os.environ:
        del os.environ["EYEPOP_URL"]
    if "EYEPOP_API_KEY" in os.environ:
        del os.environ["EYEPOP_API_KEY"]

    if original_url is not None:
        os.environ["EYEPOP_URL"] = original_url
    if original_key is not None:
        os.environ["EYEPOP_API_KEY"] = original_key


@pytest.mark.asyncio
async def test_integrates_successfully_with_compute_api(aioresponses, clean_environment):
    """Test successful integration with compute API."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[MOCK_SESSION_RESPONSE],
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )

    async with aiohttp.ClientSession() as session:
        session_response = await fetch_new_compute_session(compute_config, session)

    assert session_response.session_endpoint == "https://integration-pipeline.example.com"
    assert session_response.m2m_access_token == "jwt-token-123"
    assert session_response.pipeline_id == "pipeline-123"


@pytest.mark.asyncio
async def test_creates_session_when_none_exists(aioresponses, clean_environment):
    """Test session creation when no sessions exist."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[],
        status=200
    )
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=MOCK_SESSION_RESPONSE,
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key"
    )

    async with aiohttp.ClientSession() as session:
        session_response = await fetch_new_compute_session(compute_config, session)

    assert session_response.session_endpoint == "https://integration-pipeline.example.com"


@pytest.mark.asyncio
async def test_creates_session_when_get_returns_404(aioresponses, clean_environment):
    """Test session creation when GET returns 404 (no sessions exist)."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        status=404
    )
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=MOCK_SESSION_RESPONSE,
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key"
    )

    async with aiohttp.ClientSession() as session:
        session_response = await fetch_new_compute_session(compute_config, session)

    assert session_response.session_endpoint == "https://integration-pipeline.example.com"
    assert session_response.m2m_access_token == "jwt-token-123"
    assert session_response.session_uuid == "session-456"


@pytest.mark.asyncio
async def test_handles_network_error(aioresponses, clean_environment):
    """Test handling of network errors."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        exception=Exception("Network error")
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key"
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(Exception, match="Network error"):
            await fetch_new_compute_session(compute_config, session)


@pytest.mark.asyncio
async def test_mimics_worker_endpoint_integration(aioresponses, clean_environment):
    """Test worker endpoint integration scenario."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[MOCK_SESSION_RESPONSE],
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key"
    )

    async with aiohttp.ClientSession() as session:
        session_response = await fetch_new_compute_session(compute_config, session)

    worker_config = {
        "session_endpoint": session_response.session_endpoint,
        "pipeline_id": session_response.pipeline_id,
        "endpoints": []
    }

    assert worker_config["session_endpoint"] == "https://integration-pipeline.example.com"
    assert worker_config["pipeline_id"] == "pipeline-123"


def test_uses_environment_variables(clean_environment):
    """Test using environment variables for configuration."""
    os.environ["EYEPOP_URL"] = "https://custom.compute.com"
    os.environ["EYEPOP_API_KEY"] = "env-api-key"

    compute_config = ComputeContext()

    assert compute_config.compute_url == "https://custom.compute.com"

    compute_config_with_key = ComputeContext(api_key="env-api-key")
    assert compute_config_with_key.api_key == "env-api-key"


@pytest.mark.asyncio
async def test_handles_array_response_from_sessions(aioresponses, clean_environment):
    """Test handling of array response from /v1/sessions endpoint."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[
            {
                "session_name": "session-3b3a8b91",
                "session_endpoint": "https://sessions.staging.eyepop.xyz/3b3a8b91",
                "session_uuid": "3b3a8b91",
                "access_token": "jwt-abc-123",
                "pipelines": [],
                "session_status": "running",
                "session_active": True
            }
        ],
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-token"
    )

    async with aiohttp.ClientSession() as session:
        session_response = await fetch_new_compute_session(compute_config, session)

    assert session_response.session_endpoint == "https://sessions.staging.eyepop.xyz/3b3a8b91"
    assert session_response.m2m_access_token == "jwt-abc-123"


@pytest.mark.asyncio
async def test_handles_single_object_response(aioresponses, clean_environment):
    """Test handling of single object response."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=MOCK_SESSION_RESPONSE,  # Single object, not array
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-token"
    )

    async with aiohttp.ClientSession() as session:
        session_response = await fetch_new_compute_session(compute_config, session)

    assert session_response.session_endpoint == "https://integration-pipeline.example.com"


@pytest.mark.asyncio
async def test_raises_on_empty_response_and_failed_create(aioresponses, clean_environment):
    """Test error when no sessions exist and creation fails."""
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[],
        status=200
    )
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        exception=Exception("Creation failed")
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-token"
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException, match="failed to create"):
            await fetch_new_compute_session(compute_config, session)


@pytest.mark.asyncio
async def test_validates_access_token_presence(aioresponses, clean_environment):
    """Test validation of access_token presence."""
    response_without_token = {**MOCK_SESSION_RESPONSE}
    response_without_token["access_token"] = ""

    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[response_without_token],
        status=200
    )

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-token"
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException, match="access_token"):
            await fetch_new_compute_session(compute_config, session)


@pytest.mark.asyncio
@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
async def test_full_flow_with_health_check(mock_fetch_new, mock_wait, clean_environment):
    """Test complete flow including health check."""
    mock_context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-key",
        session_endpoint="https://session.example.com",
        m2m_access_token="jwt-123",
        pipeline_id="pipeline-456"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    async with aiohttp.ClientSession() as session:
        result = await fetch_session_endpoint(mock_context, session)

    mock_fetch_new.assert_called_once_with(mock_context, session)
    mock_wait.assert_called_once_with(mock_context, session)
    assert result.session_endpoint == "https://session.example.com"
    assert result.m2m_access_token == "jwt-123"
    assert result.pipeline_id == "pipeline-456"


@pytest.mark.asyncio
@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
async def test_fetch_session_endpoint_passes_context(mock_fetch_new, mock_wait):
    """Test that fetch_session_endpoint correctly passes context through."""
    input_context = ComputeContext(
        compute_url="https://custom.compute.com",
        api_key="custom-key"
    )
    mock_context = ComputeContext(
        compute_url="https://custom.compute.com",
        api_key="custom-key",
        session_endpoint="https://session.example.com",
        m2m_access_token="jwt-123"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    async with aiohttp.ClientSession() as session:
        await fetch_session_endpoint(input_context, session)

    called_context = mock_fetch_new.call_args[0][0]
    assert called_context.compute_url == "https://custom.compute.com"
    assert called_context.api_key == "custom-key"
