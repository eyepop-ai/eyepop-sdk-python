import aiohttp
import pytest

from eyepop.compute import ComputeContext
from eyepop.compute.api import refresh_compute_token
from eyepop.exceptions import ComputeTokenException


@pytest.fixture
def mock_compute_context():
    return ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        session_uuid="session-123",
        api_key="test-api-key",
        m2m_access_token="old-token",
        access_token_expires_at="2025-10-17T12:00:00Z",
        access_token_expires_in=3600
    )


@pytest.mark.asyncio
async def test_refresh_compute_token_success(aioresponses, mock_compute_context):
    """Test successful token refresh."""
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/auth/authenticate",
        payload={
            "access_token": "new-refreshed-token",
            "expires_at": "2025-10-18T12:00:00Z",
            "expires_in": 7200
        },
        status=200
    )

    async with aiohttp.ClientSession() as session:
        result = await refresh_compute_token(mock_compute_context, session)

    # Verify token was updated
    assert result.m2m_access_token == "new-refreshed-token"
    assert result.access_token_expires_at == "2025-10-18T12:00:00Z"
    assert result.access_token_expires_in == 7200


@pytest.mark.asyncio
async def test_refresh_compute_token_missing_api_key():
    """Test token refresh fails when api_key is missing."""
    context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        session_uuid="session-123"
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeTokenException, match="api_key"):
            await refresh_compute_token(context, session)


@pytest.mark.asyncio
async def test_refresh_compute_token_http_error(aioresponses, mock_compute_context):
    """Test token refresh handles HTTP errors."""
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/auth/authenticate",
        status=401
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeTokenException, match="refresh"):
            await refresh_compute_token(mock_compute_context, session)


@pytest.mark.asyncio
async def test_refresh_compute_token_network_error(aioresponses, mock_compute_context):
    """Test token refresh handles network errors."""
    aioresponses.post(
        "https://compute.staging.eyepop.xyz/v1/auth/authenticate",
        exception=Exception("Network error")
    )

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeTokenException, match="refresh"):
            await refresh_compute_token(mock_compute_context, session)
