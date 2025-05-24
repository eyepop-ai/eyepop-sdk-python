import os
from unittest.mock import MagicMock, patch

import pytest


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
    mock_response.json.return_value = {
        "pipeline_url": "https://integration-pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "integration-test-token"

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        pipeline_url = fetch_worker_endpoint_url_from_compute(compute_token)
        assert pipeline_url == "https://integration-pipeline.example.com"

        mock_post.assert_called_once_with(
            "https://compute-api.staging.eyepop.xyz/api/v1/session",
            headers={
                "X-Token": "integration-test-token",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=None
        )
    else:
        pytest.fail("Compute token should be detected")


@patch("eyepop.compute.api.requests.post")
def test_handles_fallback_when_compute_api_fails(mock_post, clean_environment):
    mock_post.side_effect = Exception("Network error")

    os.environ["_COMPUTE_API_TOKEN"] = "fallback-test-token"

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        with pytest.raises(Exception, match="Network error"):
            fetch_worker_endpoint_url_from_compute(compute_token)

        mock_post.assert_called_once_with(
            "https://compute-api.staging.eyepop.xyz/api/v1/session",
            headers={
                "X-Token": "fallback-test-token",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=None
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
    mock_response.json.return_value = {
        "pipeline_url": "https://worker-endpoint-pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "worker-integration-token"

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        worker_url = fetch_worker_endpoint_url_from_compute(compute_token)
        if worker_url:
            worker_config = {
                "base_url": worker_url,
                "pipeline_id": "compute",
                "status": "active_prod",
            }
            is_dev_mode = False

            assert worker_config["base_url"] == "https://worker-endpoint-pipeline.example.com"
            assert worker_config["pipeline_id"] == "compute"
            assert worker_config["status"] == "active_prod"
            assert is_dev_mode is False

            mock_post.assert_called_once_with(
                "https://compute-api.staging.eyepop.xyz/api/v1/session",
                headers={
                    "X-Token": "worker-integration-token",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json=None
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
    mock_response.json.return_value = {
        "pipeline_url": "https://account-specific-pipeline.example.com",
        "pipeline_uuid": "pipeline-account-123",
        "session_uuid": "session-account-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "account-test-token"

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        pipeline_url = fetch_worker_endpoint_url_from_compute(compute_token, "account-uuid-123")
        assert pipeline_url == "https://account-specific-pipeline.example.com"

        mock_post.assert_called_once_with(
            "https://compute-api.staging.eyepop.xyz/api/v1/session",
            headers={
                "X-Token": "account-test-token",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json={"account_uuid": "account-uuid-123"}
        )
    else:
        pytest.fail("Compute token should be detected")


@patch("eyepop.compute.api.requests.post")
def test_extracts_pipeline_url_from_session_response(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://extracted-pipeline.example.com/v1/sessions/abc",
        "pipeline_uuid": "uuid-extracted-123",
        "session_uuid": "session-extracted-456",
        "status": "pending"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    os.environ["_COMPUTE_API_TOKEN"] = "extraction-test-token"

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    compute_token = os.getenv("_COMPUTE_API_TOKEN")
    if compute_token:
        pipeline_url = fetch_worker_endpoint_url_from_compute(compute_token)
        assert pipeline_url == "https://extracted-pipeline.example.com/v1/sessions/abc"
    else:
        pytest.fail("Compute token should be detected")


@patch("eyepop.compute.api.requests.post")
def test_handles_different_pipeline_statuses_in_integration(mock_post, clean_environment):
    os.environ["_COMPUTE_API_TOKEN"] = "status-test-token"
    
    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    statuses = ["unknown", "pending", "running", "stopped", "failed"]
    
    for status in statuses:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "pipeline_url": f"https://pipeline-{status}.example.com",
            "pipeline_uuid": f"pipeline-{status}-123",
            "session_uuid": f"session-{status}-456",
            "status": status
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        compute_token = os.getenv("_COMPUTE_API_TOKEN")
        if compute_token:
            pipeline_url = fetch_worker_endpoint_url_from_compute(compute_token)
            assert pipeline_url == f"https://pipeline-{status}.example.com"
        else:
            pytest.fail("Compute token should be detected")
        
        mock_post.reset_mock()