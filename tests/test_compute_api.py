from unittest.mock import patch

import aiohttp
import pytest

from eyepop.compute import ComputeContext
from eyepop.compute.api import fetch_new_compute_session, fetch_session_endpoint
from eyepop.exceptions import ComputeSessionException

MOCK_SESSION_RESPONSE = {
    "session_uuid": "session-456",
    "session_endpoint": "https://pipeline.example.com",
    "access_token": "jwt-token-123",
    "pipelines": [{"pipeline_id": "pipeline-123"}],
    "pipeline_uuid": "pipeline-123",
    "pipeline_version": "1.0.0",
    "session_status": "running",
    "session_message": "Session created successfully",
    "pipeline_ttl": 3600,
    "session_active": True
}

MOCK_SESSION_RESPONSE_NO_PIPELINES = {
    **MOCK_SESSION_RESPONSE,
    "pipelines": []
}


@pytest.fixture
def mock_compute_config():
    return ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key"
    )


@pytest.mark.asyncio
async def test_fetches_existing_session_successfully(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[MOCK_SESSION_RESPONSE],
        status=200
    )

    async with aiohttp.ClientSession() as session:
        result = await fetch_new_compute_session(mock_compute_config, session)

    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.m2m_access_token == "jwt-token-123"
    assert result.pipeline_id == "pipeline-123"


@pytest.mark.asyncio
async def test_creates_session_when_none_exists(mock_compute_config, aioresponses):
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

    async with aiohttp.ClientSession() as session:
        result = await fetch_new_compute_session(mock_compute_config, session)

    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.m2m_access_token == "jwt-token-123"


@pytest.mark.asyncio
async def test_creates_session_when_get_returns_404(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        status=404
    )
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=MOCK_SESSION_RESPONSE,
        status=200
    )

    async with aiohttp.ClientSession() as session:
        result = await fetch_new_compute_session(mock_compute_config, session)

    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.m2m_access_token == "jwt-token-123"


@pytest.mark.asyncio
async def test_handles_single_session_response(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=MOCK_SESSION_RESPONSE,
        status=200
    )

    async with aiohttp.ClientSession() as session:
        result = await fetch_new_compute_session(mock_compute_config, session)

    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.m2m_access_token == "jwt-token-123"


@pytest.mark.asyncio
async def test_handles_empty_pipelines_list(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[MOCK_SESSION_RESPONSE_NO_PIPELINES],
        status=200
    )

    async with aiohttp.ClientSession() as session:
        result = await fetch_new_compute_session(mock_compute_config, session)

    assert result.pipeline_id == ""


@pytest.mark.asyncio
async def test_raises_exception_when_no_access_token(mock_compute_config, aioresponses):
    response_without_token = {**MOCK_SESSION_RESPONSE}
    response_without_token["access_token"] = ""

    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[response_without_token],
        status=200
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException, match="access_token"):
            await fetch_new_compute_session(mock_compute_config, session)


@pytest.mark.asyncio
async def test_raises_exception_on_http_error(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        status=500
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException):
            await fetch_new_compute_session(mock_compute_config, session)


@pytest.mark.asyncio
async def test_raises_exception_on_403_forbidden(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        status=403
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException):
            await fetch_new_compute_session(mock_compute_config, session)


@pytest.mark.asyncio
async def test_raises_exception_when_post_fails(mock_compute_config, aioresponses):
    aioresponses.get(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        payload=[],
        status=200
    )
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        exception=Exception("Failed to create session")
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException, match="failed to create"):
            await fetch_new_compute_session(mock_compute_config, session)


@pytest.mark.asyncio
@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
async def test_fetch_session_endpoint_with_health_check(mock_fetch_new, mock_wait):
    mock_context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-key",
        session_endpoint="https://session.example.com",
        m2m_access_token="jwt-123"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    async with aiohttp.ClientSession() as session:
        result = await fetch_session_endpoint(mock_context, session)

    mock_fetch_new.assert_called_once_with(mock_context, session)
    mock_wait.assert_called_once_with(mock_context, session)
    assert result == mock_context


@pytest.mark.asyncio
@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
async def test_fetch_session_endpoint_raises_on_health_check_failure(mock_fetch_new, mock_wait):
    mock_context = ComputeContext()
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = False

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeSessionException, match="Failed to fetch"):
            await fetch_session_endpoint(mock_context, session)


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
