import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError


@pytest.fixture
def clean_environment():
    original_compute_url = os.environ.get("_COMPUTE_API_URL")
    if "_COMPUTE_API_URL" in os.environ:
        del os.environ["_COMPUTE_API_URL"]

    import importlib

    import eyepop.compute.api

    importlib.reload(eyepop.compute.api)

    yield

    if "_COMPUTE_API_URL" in os.environ:
        del os.environ["_COMPUTE_API_URL"]
    if original_compute_url is not None:
        os.environ["_COMPUTE_API_URL"] = original_compute_url

    importlib.reload(eyepop.compute.api)


@patch("eyepop.compute.api.requests.post")
def test_fetches_pipeline_url_successfully(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    result = fetch_worker_endpoint_url_from_compute("test-token")

    mock_post.assert_called_once_with(
        "https://compute-api.staging.eyepop.xyz/api/v1/session",
        headers={
            "X-Token": "test-token",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=None
    )
    assert result == "https://pipeline.example.com"


@patch("eyepop.compute.api.requests.post")
def test_includes_account_uuid_in_request_body(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    result = fetch_worker_endpoint_url_from_compute("test-token", "account-uuid-789")

    mock_post.assert_called_once_with(
        "https://compute-api.staging.eyepop.xyz/api/v1/session",
        headers={
            "X-Token": "test-token",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"account_uuid": "account-uuid-789"}
    )
    assert result == "https://pipeline.example.com"


@patch("eyepop.compute.api.requests.post")
def test_uses_custom_api_url_when_set(mock_post, clean_environment):
    os.environ["_COMPUTE_API_URL"] = "https://custom-api.example.com"

    import importlib

    import eyepop.compute.api

    importlib.reload(eyepop.compute.api)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://custom-pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    result = fetch_worker_endpoint_url_from_compute("custom-token")

    mock_post.assert_called_once_with(
        "https://custom-api.example.com/api/v1/session",
        headers={
            "X-Token": "custom-token",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=None
    )
    assert result == "https://custom-pipeline.example.com"


@patch("eyepop.compute.api.requests.post")
def test_validates_response_with_pydantic(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "pending"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    result = fetch_worker_endpoint_url_from_compute("test-token")

    assert result == "https://pipeline.example.com"


@patch("eyepop.compute.api.requests.post")
def test_raises_validation_error_on_invalid_response(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://pipeline.example.com",
        "status": "invalid_status"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    with pytest.raises(ValidationError):
        fetch_worker_endpoint_url_from_compute("test-token")


@patch("eyepop.compute.api.requests.post")
def test_raises_exception_on_http_error(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 404 Not Found")
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    with pytest.raises(Exception, match="HTTP 404"):
        fetch_worker_endpoint_url_from_compute("test-token")


@patch("eyepop.compute.api.requests.post")
def test_raises_exception_on_connection_error(mock_post, clean_environment):
    mock_post.side_effect = Exception("Connection refused")

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    with pytest.raises(Exception, match="Connection refused"):
        fetch_worker_endpoint_url_from_compute("test-token")


@patch("eyepop.compute.api.requests.post")
def test_raises_exception_on_json_decode_error(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.side_effect = ValueError("Invalid JSON")
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    with pytest.raises(ValueError, match="Invalid JSON"):
        fetch_worker_endpoint_url_from_compute("test-token")


@patch("eyepop.compute.api.requests.post")
def test_handles_different_pipeline_statuses(mock_post, clean_environment):
    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    statuses = ["unknown", "pending", "running", "stopped", "failed"]
    
    for status in statuses:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "pipeline_url": f"https://pipeline-{status}.example.com",
            "pipeline_uuid": f"pipeline-{status}",
            "session_uuid": f"session-{status}",
            "status": status
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = fetch_worker_endpoint_url_from_compute("test-token")
        assert result == f"https://pipeline-{status}.example.com"
        mock_post.reset_mock()


@patch("eyepop.compute.api.requests.post")
def test_passes_different_tokens_correctly(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    tokens = ["token1", "very-long-token-with-special-chars-123!@#", "short", ""]

    for token in tokens:
        mock_post.reset_mock()
        result = fetch_worker_endpoint_url_from_compute(token)

        mock_post.assert_called_once_with(
            "https://compute-api.staging.eyepop.xyz/api/v1/session",
            headers={
                "X-Token": token,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=None
        )
        assert result == "https://pipeline.example.com"


def test_respects_compute_api_url_environment_variable(clean_environment):
    import eyepop.compute.api

    default_url = eyepop.compute.api._compute_url
    assert default_url == "https://compute-api.staging.eyepop.xyz"

    os.environ["_COMPUTE_API_URL"] = "https://my-custom-compute.example.com"

    import importlib

    importlib.reload(eyepop.compute.api)

    custom_url = eyepop.compute.api._compute_url
    assert custom_url == "https://my-custom-compute.example.com"


@patch("eyepop.compute.api.requests.post")
def test_extracts_pipeline_url_from_session_response(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "pipeline_url": "https://specific-pipeline.example.com/v2/pipeline",
        "pipeline_uuid": "uuid-123-456",
        "session_uuid": "session-789-012",
        "status": "running"
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    from eyepop.compute.api import fetch_worker_endpoint_url_from_compute

    result = fetch_worker_endpoint_url_from_compute("test-token")

    assert result == "https://specific-pipeline.example.com/v2/pipeline"