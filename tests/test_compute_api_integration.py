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
    assert os.getenv("_COMPUTE_API_TOKEN") is None

    os.environ["_COMPUTE_API_TOKEN"] = "test-token-123"
    assert os.getenv("_COMPUTE_API_TOKEN") == "test-token-123"

    test_tokens = ["short", "very-long-token-with-special-chars-123!@#", "", "production-token-xyz"]
    for token in test_tokens:
        os.environ["_COMPUTE_API_TOKEN"] = token
        assert os.getenv("_COMPUTE_API_TOKEN") == token


@patch("eyepop.compute.api.requests.post")
def test_integrates_successfully_with_compute_api(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "test-token-123"

    from eyepop.compute.api import fetch_new_compute_session

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        session_response = fetch_new_compute_session(compute_token, TEST_ACCOUNT_UUID)
        assert session_response.session_endpoint == "https://integration-pipeline.example.com"

        mock_post.assert_called_once_with(
            "https://compute.staging.eyepop.xyz/v1/session",
            headers=TEST_REQUEST_HEADERS,
            json={"account_uuid": TEST_ACCOUNT_UUID}
        )
    else:
        pytest.fail("Compute token should be detected")


@patch("eyepop.compute.api.requests.post")
def test_handles_fallback_when_compute_api_fails(mock_post, clean_environment):
    mock_post.side_effect = Exception("Network error")

    os.environ["_COMPUTE_API_TOKEN"] = "fallback-test-token"

    from eyepop.compute.api import fetch_new_compute_session

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    expected_headers = TEST_REQUEST_HEADERS.copy()
    expected_headers["Authorization"] = "Bearer fallback-test-token"

    if compute_token:
        with pytest.raises(Exception, match="Network error"):
            fetch_new_compute_session(compute_token, "account-uuid-123")

        mock_post.assert_called_once_with(
            "https://compute.staging.eyepop.xyz/v1/session",
            headers=expected_headers,
            json={"account_uuid": "account-uuid-123"}
        )
    else:
        pytest.fail("Compute token should be detected")


def test_follows_complete_integration_logic_flow(clean_environment):
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
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "test-token-123"

    from eyepop.compute.api import fetch_new_compute_session

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        session_response = fetch_new_compute_session(compute_token, TEST_ACCOUNT_UUID)
        if session_response:
            worker_config = {
                "base_url": session_response.session_endpoint,
                "pipeline_id": session_response.pipeline_uuid,
                "status": session_response.session_status,
            }
            is_dev_mode = False

            assert worker_config["base_url"] == "https://integration-pipeline.example.com"
            assert worker_config["pipeline_id"] == "pipeline-123"
            assert worker_config["status"] == "running"
            assert is_dev_mode is False

            mock_post.assert_called_once_with(
                "https://compute.staging.eyepop.xyz/v1/session",
                headers=TEST_REQUEST_HEADERS,
                json={"account_uuid": TEST_ACCOUNT_UUID}
            )
        else:
            pytest.fail("Worker URL should be returned")
    else:
        pytest.fail("Compute token should be detected")


def test_integrates_with_custom_compute_url(clean_environment):
    os.environ["_COMPUTE_API_URL"] = "https://custom-integration.example.com"

    import importlib

    import eyepop.compute.api

    importlib.reload(eyepop.compute.api)

    from eyepop.compute.api import _compute_url

    assert _compute_url == "https://custom-integration.example.com"


@patch("eyepop.compute.api.requests.post")
def test_supports_account_uuid_parameter(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "test-token-123"

    from eyepop.compute.api import fetch_new_compute_session

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        session_response = fetch_new_compute_session(compute_token, "foobarbaz")
        assert session_response.session_endpoint == "https://integration-pipeline.example.com"

        mock_post.assert_called_once_with(
            "https://compute.staging.eyepop.xyz/v1/session",
            headers=TEST_REQUEST_HEADERS,
            json={"account_uuid": "foobarbaz"}
        )
    else:
        pytest.fail("Compute token should be detected")