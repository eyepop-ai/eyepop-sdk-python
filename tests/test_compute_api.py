import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from eyepop.compute.api import fetch_new_compute_session, fetch_session_endpoint
from eyepop.compute.models import ComputeContext

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

TEST_REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer test-secret-key"
}


@pytest.fixture
def mock_compute_config():
    return ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-secret-key"
    )


@patch("eyepop.compute.api.requests.get")
def test_fetches_existing_session_successfully(mock_get, mock_compute_config):
    mock_response = MagicMock()
    mock_response.json.return_value = [MOCK_SESSION_RESPONSE]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_new_compute_session(mock_compute_config)

    mock_get.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )
    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.access_token == "jwt-token-123"
    assert result.pipeline_id == "pipeline-123"


@patch("eyepop.compute.api.requests.post")
@patch("eyepop.compute.api.requests.get")
def test_creates_session_when_none_exists(mock_get, mock_post, mock_compute_config):
    mock_get_response = MagicMock()
    mock_get_response.json.return_value = []
    mock_get_response.raise_for_status.return_value = None
    mock_get.return_value = mock_get_response

    mock_post_response = MagicMock()
    mock_post_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_post_response.raise_for_status.return_value = None
    mock_post.return_value = mock_post_response

    result = fetch_new_compute_session(mock_compute_config)

    mock_get.assert_called_once()
    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )
    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.access_token == "jwt-token-123"

@patch("eyepop.compute.api.requests.post")
@patch("eyepop.compute.api.requests.get")
def test_creates_session_when_get_returns_404(mock_get, mock_post, mock_compute_config):
    # Mock GET request to return 404
    mock_get_response = MagicMock()
    mock_http_error = requests.HTTPError()
    mock_http_error.response = MagicMock()
    mock_http_error.response.status_code = 404
    mock_get_response.raise_for_status.side_effect = mock_http_error
    mock_get.return_value = mock_get_response

    # Mock POST request to succeed
    mock_post_response = MagicMock()
    mock_post_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_post_response.raise_for_status.return_value = None
    mock_post.return_value = mock_post_response

    result = fetch_new_compute_session(mock_compute_config)

    mock_get.assert_called_once()
    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )
    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.access_token == "jwt-token-123"


@patch("eyepop.compute.api.requests.get")
def test_handles_single_session_response(mock_get, mock_compute_config):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_new_compute_session(mock_compute_config)

    assert result.session_endpoint == "https://pipeline.example.com"
    assert result.access_token == "jwt-token-123"


@patch("eyepop.compute.api.requests.get")
def test_handles_empty_pipelines_list(mock_get, mock_compute_config):
    mock_response = MagicMock()
    mock_response.json.return_value = [MOCK_SESSION_RESPONSE_NO_PIPELINES]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = fetch_new_compute_session(mock_compute_config)

    assert result.pipeline_id == ""


@patch("eyepop.compute.api.requests.get")
def test_raises_exception_when_no_access_token(mock_get, mock_compute_config):
    response_without_token = {**MOCK_SESSION_RESPONSE}
    response_without_token["access_token"] = ""
    
    mock_response = MagicMock()
    mock_response.json.return_value = [response_without_token]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    with pytest.raises(Exception, match="No access_token received"):
        fetch_new_compute_session(mock_compute_config)


@patch("eyepop.compute.api.requests.get")
def test_raises_exception_on_http_error(mock_get, mock_compute_config):
    # Test that non-404 HTTP errors are properly raised
    mock_response = MagicMock()
    mock_http_error = requests.HTTPError()
    mock_http_error.response = MagicMock()
    mock_http_error.response.status_code = 500  # Internal Server Error
    mock_response.raise_for_status.side_effect = mock_http_error
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_new_compute_session(mock_compute_config)

@patch("eyepop.compute.api.requests.get")
def test_raises_exception_on_403_forbidden(mock_get, mock_compute_config):
    # Test that 403 Forbidden errors are properly raised
    mock_response = MagicMock()
    mock_http_error = requests.HTTPError()
    mock_http_error.response = MagicMock()
    mock_http_error.response.status_code = 403  # Forbidden
    mock_response.raise_for_status.side_effect = mock_http_error
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        fetch_new_compute_session(mock_compute_config)


@patch("eyepop.compute.api.requests.post")
@patch("eyepop.compute.api.requests.get")
def test_raises_exception_when_post_fails(mock_get, mock_post, mock_compute_config):
    mock_get_response = MagicMock()
    mock_get_response.json.return_value = []
    mock_get_response.raise_for_status.return_value = None
    mock_get.return_value = mock_get_response

    mock_post.side_effect = Exception("Failed to create session")

    with pytest.raises(Exception, match="No existing session and failed to create new one"):
        fetch_new_compute_session(mock_compute_config)


@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
def test_fetch_session_endpoint_with_health_check(mock_fetch_new, mock_wait):
    mock_context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-key",
        session_endpoint="https://session.example.com",
        access_token="jwt-123"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    result = fetch_session_endpoint(mock_context)

    mock_fetch_new.assert_called_once_with(mock_context)
    mock_wait.assert_called_once_with(mock_context)
    assert result == mock_context


@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
def test_fetch_session_endpoint_raises_on_health_check_failure(mock_fetch_new, mock_wait):
    mock_context = ComputeContext()
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = False

    with pytest.raises(Exception, match="Failed to fetch session endpoint"):
        fetch_session_endpoint(mock_context)


@patch.dict(os.environ, {"EYEPOP_URL": "https://custom.compute.com", "EYEPOP_SECRET_KEY": "env-key"})
@patch("eyepop.compute.api.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
def test_fetch_session_endpoint_uses_env_vars(mock_fetch_new, mock_wait):
    mock_context = ComputeContext(
        compute_url="https://custom.compute.com",
        secret_key="env-key"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    result = fetch_session_endpoint()

    called_context = mock_fetch_new.call_args[0][0]
    assert called_context.compute_url == "https://custom.compute.com"
    assert called_context.secret_key == "env-key"