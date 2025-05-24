import pytest
from pydantic import ValidationError

from eyepop.compute.models import ComputeApiSessionResponse, PipelineStatus


def test_creates_valid_session_response():
    """It creates a valid session response."""
    response_data = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running",
    }

    response = ComputeApiSessionResponse(**response_data)

    assert response.pipeline_url == "https://pipeline.example.com"
    assert response.pipeline_uuid == "pipeline-123"
    assert response.session_uuid == "session-456"
    assert response.status == PipelineStatus.RUNNING


def test_validates_pipeline_status_enum():
    """It only allows good enums."""
    valid_statuses = ["unknown", "pending", "running", "stopped", "failed"]

    for status in valid_statuses:
        response_data = {
            "pipeline_url": "https://pipeline.example.com",
            "pipeline_uuid": "pipeline-123",
            "session_uuid": "session-456",
            "status": status,
        }

        response = ComputeApiSessionResponse(**response_data)
        assert response.status == status

    """It rejects bad enums."""
    response_data = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "invalid_status",
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**response_data)


def test_requires_all_fields():
    incomplete_data = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**incomplete_data)


def test_validates_field_types():
    response_data = {
        "pipeline_url": 123,
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running",
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**response_data)


def test_serializes_to_dict():
    response_data = {
        "pipeline_url": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_uuid": "session-456",
        "status": "running",
    }

    response = ComputeApiSessionResponse(**response_data)
    serialized = response.model_dump()

    assert serialized == response_data


def test_parses_from_json_string():
    json_str = '{"pipeline_url": "https://pipeline.example.com", "pipeline_uuid": "pipeline-123", "session_uuid": "session-456", "status": "running"}'

    response = ComputeApiSessionResponse.model_validate_json(json_str)

    assert response.pipeline_url == "https://pipeline.example.com"
    assert response.status == PipelineStatus.RUNNING


def test_handles_different_url_formats():
    url_formats = [
        "https://simple.example.com",
        "https://complex.example.com/v1/pipelines/123",
        "http://localhost:8080/pipeline",
        "https://api.example.com/compute/v2/sessions/abc-def",
    ]

    for url in url_formats:
        response_data = {
            "pipeline_url": url,
            "pipeline_uuid": "pipeline-123",
            "session_uuid": "session-456",
            "status": "running",
        }

        response = ComputeApiSessionResponse(**response_data)
        assert response.pipeline_url == url


def test_handles_uuid_formats():
    uuid_formats = [
        "simple-uuid",
        "12345678-1234-1234-1234-123456789012",
        "abc-def-ghi",
        "session_123_456",
    ]

    for uuid_val in uuid_formats:
        response_data = {
            "pipeline_url": "https://pipeline.example.com",
            "pipeline_uuid": uuid_val,
            "session_uuid": uuid_val,
            "status": "running",
        }

        response = ComputeApiSessionResponse(**response_data)
        assert response.pipeline_uuid == uuid_val
        assert response.session_uuid == uuid_val
