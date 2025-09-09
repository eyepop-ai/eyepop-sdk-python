import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from eyepop.compute.api import fetch_new_compute_session, fetch_session_endpoint
from eyepop.compute.models import ComputeContext

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

TEST_REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer test-secret-key"
}


@pytest.fixture
def clean_environment():
    """Clean environment before and after tests"""
    original_url = os.environ.get("EYEPOP_URL")
    original_key = os.environ.get("EYEPOP_SECRET_KEY")

    if "EYEPOP_URL" in os.environ:
        del os.environ["EYEPOP_URL"]
    if "EYEPOP_SECRET_KEY" in os.environ:
        del os.environ["EYEPOP_SECRET_KEY"]

    yield

    if "EYEPOP_URL" in os.environ:
        del os.environ["EYEPOP_URL"]
    if "EYEPOP_SECRET_KEY" in os.environ:
        del os.environ["EYEPOP_SECRET_KEY"]

    if original_url is not None:
        os.environ["EYEPOP_URL"] = original_url
    if original_key is not None:
        os.environ["EYEPOP_SECRET_KEY"] = original_key


@patch("eyepop.compute.api.requests.get")
def test_integrates_successfully_with_compute_api(mock_get, clean_environment):
    """Test successful integration with compute API"""
    mock_response = MagicMock()
    mock_response.json.return_value = [MOCK_SESSION_RESPONSE]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-secret-key",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"
    assert session_response.access_token == "jwt-token-123"
    assert session_response.pipeline_id == "pipeline-123"

    mock_get.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )


@patch("eyepop.compute.api.requests.post")
@patch("eyepop.compute.api.requests.get")
def test_creates_session_when_none_exists(mock_get, mock_post, clean_environment):
    """Test session creation when no sessions exist"""
    mock_get_response = MagicMock()
    mock_get_response.json.return_value = []
    mock_get_response.raise_for_status.return_value = None
    mock_get.return_value = mock_get_response

    mock_post_response = MagicMock()
    mock_post_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_post_response.raise_for_status.return_value = None
    mock_post.return_value = mock_post_response

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-secret-key"
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"

    mock_get.assert_called_once()
    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )

@patch("eyepop.compute.api.requests.post")
@patch("eyepop.compute.api.requests.get")
def test_creates_session_when_get_returns_404(mock_get, mock_post, clean_environment):
    """Test session creation when GET returns 404 (no sessions exist)"""
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

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-secret-key"
    )
    
    # Should create a new session despite 404
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"
    assert session_response.access_token == "jwt-token-123"
    assert session_response.session_uuid == "session-456"

    # Verify both GET and POST were called
    mock_get.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )
    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/sessions",
        headers=TEST_REQUEST_HEADERS
    )


@patch("eyepop.compute.api.requests.get")
def test_handles_network_error(mock_get, clean_environment):
    """Test handling of network errors"""
    mock_get.side_effect = Exception("Network error")

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-secret-key"
    )

    with pytest.raises(Exception, match="Network error"):
        fetch_new_compute_session(compute_config)


@patch("eyepop.compute.api.requests.get")
def test_mimics_worker_endpoint_integration(mock_get, clean_environment):
    """Test worker endpoint integration scenario"""
    mock_response = MagicMock()
    mock_response.json.return_value = [MOCK_SESSION_RESPONSE]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-secret-key"
    )
    
    session_response = fetch_new_compute_session(compute_config)
    
    # Simulate worker endpoint logic
    worker_config = {
        "session_endpoint": session_response.session_endpoint,
        "pipeline_id": session_response.pipeline_id,
        "endpoints": []
    }

    assert worker_config["session_endpoint"] == "https://integration-pipeline.example.com"
    assert worker_config["pipeline_id"] == "pipeline-123"


def test_uses_environment_variables(clean_environment):
    """Test using environment variables for configuration"""
    os.environ["EYEPOP_URL"] = "https://custom.compute.com"
    os.environ["EYEPOP_SECRET_KEY"] = "env-secret-key"

    compute_config = ComputeContext()
    
    # Note: ComputeContext doesn't automatically use env vars in __init__
    # The fetch_session_endpoint function uses them
    assert compute_config.compute_url == "https://compute-api.staging.eyepop.xyz"  # Default
    assert compute_config.secret_key == "env-secret-key"  # From env


@patch("eyepop.compute.api.requests.get")
def test_handles_array_response_from_sessions(mock_get, clean_environment):
    """Test handling of array response from /v1/sessions endpoint"""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "session_name": "session-3b3a8b91",
            "session_endpoint": "https://sessions.staging.eyepop.xyz/3b3a8b91",
            "session_uuid": "3b3a8b91",
            "access_token": "jwt-abc-123",
            "pipelines": [],
            "session_status": "running",
            "session_active": True
        }
    ]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-token"
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://sessions.staging.eyepop.xyz/3b3a8b91"
    assert session_response.access_token == "jwt-abc-123"


@patch("eyepop.compute.api.requests.get")
def test_handles_single_object_response(mock_get, clean_environment):
    """Test handling of single object response"""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE  # Single object, not array
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-token"
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"


@patch("eyepop.compute.api.requests.post")
@patch("eyepop.compute.api.requests.get")
def test_raises_on_empty_response_and_failed_create(mock_get, mock_post, clean_environment):
    """Test error when no sessions exist and creation fails"""
    mock_get_response = MagicMock()
    mock_get_response.json.return_value = []
    mock_get_response.raise_for_status.return_value = None
    mock_get.return_value = mock_get_response

    mock_post.side_effect = Exception("Creation failed")

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-token"
    )
    
    with pytest.raises(Exception, match="No existing session and failed to create new one"):
        fetch_new_compute_session(compute_config)


@patch("eyepop.compute.api.requests.get")
def test_validates_access_token_presence(mock_get, clean_environment):
    """Test validation of access_token presence"""
    response_without_token = {**MOCK_SESSION_RESPONSE}
    response_without_token["access_token"] = ""
    
    mock_response = MagicMock()
    mock_response.json.return_value = [response_without_token]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-token"
    )
    
    with pytest.raises(Exception, match="No access_token received"):
        fetch_new_compute_session(compute_config)


@patch("eyepop.compute.status.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
def test_full_flow_with_health_check(mock_fetch_new, mock_wait, clean_environment):
    """Test complete flow including health check"""
    mock_context = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        secret_key="test-key",
        session_endpoint="https://session.example.com",
        access_token="jwt-123",
        pipeline_id="pipeline-456"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    result = fetch_session_endpoint(mock_context)

    mock_fetch_new.assert_called_once_with(mock_context)
    mock_wait.assert_called_once_with(mock_context)
    assert result.session_endpoint == "https://session.example.com"
    assert result.access_token == "jwt-123"
    assert result.pipeline_id == "pipeline-456"


@patch.dict(os.environ, {"EYEPOP_URL": "https://custom.compute.com", "EYEPOP_SECRET_KEY": "env-key"})
@patch("eyepop.compute.status.wait_for_session")
@patch("eyepop.compute.api.fetch_new_compute_session")
def test_fetch_session_endpoint_with_env_vars(mock_fetch_new, mock_wait):
    """Test fetch_session_endpoint using environment variables"""
    mock_context = ComputeContext(
        compute_url="https://custom.compute.com",
        secret_key="env-key",
        session_endpoint="https://session.example.com",
        access_token="jwt-123"
    )
    mock_fetch_new.return_value = mock_context
    mock_wait.return_value = True

    # Call without arguments to use env vars
    result = fetch_session_endpoint()

    # Verify it created context with env vars
    called_context = mock_fetch_new.call_args[0][0]
    assert called_context.compute_url == "https://custom.compute.com"
    assert called_context.secret_key == "env-key"