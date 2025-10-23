from unittest.mock import MagicMock, patch

import pytest

from eyepop.compute.api import refresh_compute_token
from eyepop.compute.models import ComputeContext


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


@patch("eyepop.compute.api.requests.post")
def test_refresh_compute_token_success(mock_post, mock_compute_context):
    """Test successful token refresh"""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new-refreshed-token",
        "access_token_expires_at": "2025-10-18T12:00:00Z",
        "access_token_expires_in": 7200
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = refresh_compute_token(mock_compute_context)

    # Verify correct endpoint was called
    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions/session-123/token",
        headers={
            "Authorization": "Bearer test-api-key",
            "Accept": "application/json"
        }
    )

    # Verify token was updated
    assert result.access_token == "new-refreshed-token"
    assert result.access_token_expires_at == "2025-10-18T12:00:00Z"
    assert result.access_token_expires_in == 7200


@patch("eyepop.compute.api.requests.post")
def test_refresh_compute_token_missing_session_uuid(mock_post):
    """Test token refresh fails when session_uuid is missing"""
    context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        api_key="test-api-key"
    )

    with pytest.raises(Exception, match="Cannot refresh token: no session_uuid"):
        refresh_compute_token(context)

    mock_post.assert_not_called()


@patch("eyepop.compute.api.requests.post")
def test_refresh_compute_token_missing_api_key(mock_post):
    """Test token refresh fails when api_key is missing"""
    context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        session_uuid="session-123"
    )

    with pytest.raises(Exception, match="Cannot refresh token: no api_key"):
        refresh_compute_token(context)

    mock_post.assert_not_called()


@patch("eyepop.compute.api.requests.post")
def test_refresh_compute_token_http_error(mock_post, mock_compute_context):
    """Test token refresh handles HTTP errors"""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 401 Unauthorized")
    mock_post.return_value = mock_response

    with pytest.raises(Exception, match="Token refresh failed"):
        refresh_compute_token(mock_compute_context)


@patch("eyepop.compute.api.requests.post")
def test_refresh_compute_token_network_error(mock_post, mock_compute_context):
    """Test token refresh handles network errors"""
    mock_post.side_effect = Exception("Network error")

    with pytest.raises(Exception, match="Token refresh failed"):
        refresh_compute_token(mock_compute_context)
