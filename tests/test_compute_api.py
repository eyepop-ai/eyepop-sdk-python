import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from eyepop.compute.api import fetch_new_compute_session

MOCK_SESSION_RESPONSE = {
    "session_uuid": "session-456",
    "session_endpoint": "https://pipeline.example.com",
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
    "Authorization": "Bearer test-token"
}

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
def test_fetches_session_successfully(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    session = fetch_new_compute_session("test-token", "account-uuid-123")

    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=TEST_REQUEST_HEADERS,
        json={"account_uuid": "account-uuid-123"}
    )
    assert session.session_endpoint == "https://pipeline.example.com"
    assert session.session_status == "running"
    assert session.session_active is True


@patch("eyepop.compute.api.requests.post")
def test_includes_account_uuid_in_request_body(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    session = fetch_new_compute_session("test-token", "account-uuid-789")

    mock_post.assert_called_once_with(
        "https://compute.staging.eyepop.xyz/v1/session",
        headers=TEST_REQUEST_HEADERS,
        json={"account_uuid": "account-uuid-789"}
    )
    assert session.session_endpoint == "https://pipeline.example.com"


@patch("eyepop.compute.api.requests.post")
def test_uses_custom_api_url_when_set(mock_post, clean_environment):
    os.environ["_COMPUTE_API_URL"] = "https://custom-api.example.com"

    import importlib
    import eyepop.compute.api
    importlib.reload(eyepop.compute.api)
    from eyepop.compute.api import fetch_new_compute_session, _compute_url

    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    session = fetch_new_compute_session("test-token", "account-uuid-123")

    mock_post.assert_called_once_with(
        "https://custom-api.example.com/v1/session",
        headers=TEST_REQUEST_HEADERS,
        json={"account_uuid": "account-uuid-123"}
    )
    assert session.session_endpoint == "https://pipeline.example.com"
    assert _compute_url == "https://custom-api.example.com"


@patch("eyepop.compute.api.requests.post")
def test_validates_response_with_pydantic(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_SESSION_RESPONSE
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    session = fetch_new_compute_session("test-token", "account-uuid-123")
    assert session.session_endpoint == "https://pipeline.example.com"
    assert session.session_status == "running"


@patch("eyepop.compute.api.requests.post")
def test_raises_exception_on_http_error(mock_post, clean_environment):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 404 Not Found")
    mock_post.return_value = mock_response

    with pytest.raises(Exception, match="HTTP 404"):
        fetch_new_compute_session("test-token", "account-uuid-123")


@patch("eyepop.compute.api.requests.post")
def test_raises_exception_on_connection_error(mock_post, clean_environment):
    mock_post.side_effect = Exception("Connection refused")

    with pytest.raises(Exception, match="Connection refused"):
        fetch_new_compute_session("test-token", "account-uuid-123")


def test_respects_compute_api_url_environment_variable(clean_environment):
    import eyepop.compute.api

    default_url = eyepop.compute.api._compute_url
    assert default_url == "https://compute.staging.eyepop.xyz"

    os.environ["_COMPUTE_API_URL"] = "https://my-custom-compute.example.com"

    import importlib
    importlib.reload(eyepop.compute.api)

    custom_url = eyepop.compute.api._compute_url
    assert custom_url == "https://my-custom-compute.example.com"