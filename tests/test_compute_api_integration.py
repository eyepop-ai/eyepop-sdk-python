import os
from unittest.mock import MagicMock, patch

import pytest

TEST_ACCOUNT_UUID = "test-account-uuid-123"

TEST_REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Authorization": "Bearer test-token-123"
}

MOCK_SESSION_RESPONSE = {
    "session_uuid": "session-456",
    "session_endpoint": "https://integration-pipeline.example.com",
    "pipeline_uuid": "pipeline-123",
    "pipeline_version": "1.0.0",
    "session_status": "running",
    "session_message": "Session created successfully",
    "pipeline_ttl": 3600,
    "session_active": True
}

@pytest.fixture
def clean_environment():
    """Clean environment before and after tests"""
    original_compute_token = os.environ.get("_COMPUTE_API_TOKEN")
    original_compute_url = os.environ.get("_COMPUTE_API_URL")

    if "_COMPUTE_API_TOKEN" in os.environ:
        del os.environ["_COMPUTE_API_TOKEN"]
    if "_COMPUTE_API_URL" in os.environ:
        del os.environ["_COMPUTE_API_URL"]

    import importlib
    import eyepop.compute.api
    importlib.reload(eyepop.compute.api)

    yield

    if "_COMPUTE_API_TOKEN" in os.environ:
        del os.environ["_COMPUTE_API_TOKEN"]
    if "_COMPUTE_API_URL" in os.environ:
        del os.environ["_COMPUTE_API_URL"]

    if original_compute_token is not None:
        os.environ["_COMPUTE_API_TOKEN"] = original_compute_token
    if original_compute_url is not None:
        os.environ["_COMPUTE_API_URL"] = original_compute_url

    importlib.reload(eyepop.compute.api)


def test_detects_environment_variables_correctly(clean_environment):
    """Test environment variable detection"""
    assert os.getenv("_COMPUTE_API_TOKEN") is None

    os.environ["_COMPUTE_API_TOKEN"] = "test-token-123"
    assert os.getenv("_COMPUTE_API_TOKEN") == "test-token-123"

    test_tokens = ["short", "very-long-token-with-special-chars-123!@#", "", "production-token-xyz"]
    for token in test_tokens:
        os.environ["_COMPUTE_API_TOKEN"] = token
        assert os.getenv("_COMPUTE_API_TOKEN") == token


@patch("eyepop.compute.api.requests.post")
def test_integrates_successfully_with_compute_api(mock_post, clean_environment):
    """Test successful integration with compute API"""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        user_uuid=TEST_ACCOUNT_UUID,
        secret_key="test-token-123",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"

    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=TEST_REQUEST_HEADERS,
        json={"user_uuid": TEST_ACCOUNT_UUID}
    )


@patch("eyepop.compute.api.requests.post")
def test_handles_fallback_when_compute_api_fails(mock_post, clean_environment):
    """Test fallback handling when compute API fails"""
    mock_post.side_effect = Exception("Network error")

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        user_uuid="account-uuid-123",
        secret_key="fallback-test-token",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    expected_headers = TEST_REQUEST_HEADERS.copy()
    expected_headers["Authorization"] = "Bearer fallback-test-token"

    with pytest.raises(Exception, match="Network error"):
        fetch_new_compute_session(compute_config)

    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=expected_headers,
        json={"user_uuid": "account-uuid-123"}
    )


def test_follows_complete_integration_logic_flow(clean_environment):
    """Test complete integration logic flow"""
    assert os.getenv("_COMPUTE_API_TOKEN") is None

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        pytest.fail("Should not detect token when none is set")

    os.environ["_COMPUTE_API_TOKEN"] = "flow-test-token"

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        assert compute_token == "flow-test-token"
    else:
        pytest.fail("Should detect token when it is set")


@patch("eyepop.compute.api.requests.post")
def test_mimics_worker_endpoint_integration_scenario(mock_post, clean_environment):
    """Test worker endpoint integration scenario"""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        user_uuid=TEST_ACCOUNT_UUID,
        secret_key="test-token-123",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    session_response = fetch_new_compute_session(compute_config)
    
    # Simulate worker endpoint logic
    worker_config = {
        "base_url": session_response.session_endpoint,
        "pipeline_id": session_response.pipeline_uuid,
        "status": "running",
    }
    is_dev_mode = False

    assert worker_config["base_url"] == "https://integration-pipeline.example.com"
    assert worker_config["pipeline_id"] == "pipeline-123"
    assert worker_config["status"] == "running"
    assert is_dev_mode is False

    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=TEST_REQUEST_HEADERS,
        json={"user_uuid": TEST_ACCOUNT_UUID}
    )


def test_integrates_with_custom_compute_url(clean_environment):
    """Test integration with custom compute URL"""
    os.environ["EYEPOP_URL"] = "https://custom-integration.example.com"

    from eyepop.compute.models import ComputeContext
    
    compute_config = ComputeContext(
        compute_url="https://custom-integration.example.com",
        user_uuid="test",
        secret_key="test",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    assert compute_config.compute_url == "https://custom-integration.example.com"


@patch("eyepop.compute.api.requests.post")
def test_supports_user_uuid_parameter(mock_post, clean_environment):
    """Test support for user_uuid parameter"""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        user_uuid="foobarbaz",
        secret_key="test-token-123",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"

    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=TEST_REQUEST_HEADERS,
        json={"user_uuid": "foobarbaz"}
    )


@patch("eyepop.compute.api.requests.get")
def test_handles_array_response_from_sessions_endpoint(mock_get, clean_environment):
    """Test handling of array response from /v1/sessions endpoint"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{
        "session_name": "session-3b3a8b91",
        "session_endpoint": "https://sessions.staging.eyepop.xyz/3b3a8b91-142f-4423-bb56-e772107fbaa6",
        "session_uuid": "3b3a8b91-142f-4423-bb56-e772107fbaa6",
        "user_uuid": "ff92bf8c460f11ef8a820a359ae0bb9d",
        "session_status": "running",
        "session_active": False,
        "created_at": "2025-08-12T23:50:55Z"
    }]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    # Mock the GET request for sessions list
    with patch("eyepop.compute.api.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=405)  # Method not allowed
        
        with patch("eyepop.compute.api.requests.get", return_value=mock_response):
            compute_config = ComputeContext(
                compute_url="https://compute.staging.eyepop.xyz",
                user_uuid="test-user",
                secret_key="test-token",
                wait_for_session_timeout=30,
                wait_for_session_interval=2
            )
            
            # Since our current implementation uses POST, not GET,
            # we need to handle array response in POST
            mock_post.return_value = mock_response
            mock_post.return_value.status_code = 200
            
            session_response = fetch_new_compute_session(compute_config)
            assert session_response.session_endpoint == "https://sessions.staging.eyepop.xyz/3b3a8b91-142f-4423-bb56-e772107fbaa6"


@patch("eyepop.compute.api.requests.post")
def test_handles_empty_sessions_array(mock_post, clean_environment):
    """Test handling of empty sessions array"""
    mock_response = MagicMock()
    mock_response.json.return_value = []
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        user_uuid="test-user",
        secret_key="test-token",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    with pytest.raises(Exception, match="No sessions returned from compute API"):
        fetch_new_compute_session(compute_config)


@patch("eyepop.compute.api.requests.post")
def test_authentication_with_jwt_token(mock_post, clean_environment):
    """Test authentication using JWT token"""
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["EYEPOP_PIPELINE_JWT"] = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9..."

    from eyepop.compute.api import fetch_new_compute_session
    from eyepop.compute.models import ComputeContext

    compute_config = ComputeContext(
        compute_url="https://compute.staging.eyepop.xyz",
        user_uuid="test-user",
        secret_key="secret-key-123",
        wait_for_session_timeout=30,
        wait_for_session_interval=2
    )
    
    session_response = fetch_new_compute_session(compute_config)
    assert session_response.session_endpoint == "https://integration-pipeline.example.com"

    # Should use secret key, not JWT for compute API
    expected_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": "Bearer secret-key-123"
    }
    
    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=expected_headers,
        json={"user_uuid": "test-user"}
    )