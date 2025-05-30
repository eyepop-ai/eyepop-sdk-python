import pytest
from pydantic import ValidationError

from eyepop.compute.models import ComputeApiSessionResponse, PipelineStatus


def test_creates_valid_session_response():
    """It creates a valid session response."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "pipeline_version": "1.0.0",
        "session_status": PipelineStatus.RUNNING,
        "session_message": "Session created successfully",
        "pipeline_ttl": 3600,
        "session_active": True
    }

    response = ComputeApiSessionResponse(**response_data)

    assert response.session_endpoint == "https://pipeline.example.com"
    assert response.pipeline_uuid == "pipeline-123"
    assert response.session_uuid == "session-456"
    assert response.session_status == PipelineStatus.RUNNING
    assert response.session_active is True


def test_validates_pipeline_status_enum():
    """It only allows good enums."""
    valid_statuses = [PipelineStatus.UNKNOWN, PipelineStatus.PENDING, PipelineStatus.RUNNING, PipelineStatus.STOPPED, PipelineStatus.FAILED]

    for status in valid_statuses:
        response_data = {
            "session_uuid": "session-456",
            "session_endpoint": "https://pipeline.example.com",
            "pipeline_uuid": "pipeline-123",
            "pipeline_version": "1.0.0",
            "session_status": status,
            "session_message": "Session created successfully",
            "pipeline_ttl": 3600,
            "session_active": True
        }
        response = ComputeApiSessionResponse(**response_data)
        assert response.session_status == status

    """It rejects bad enums."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "pipeline_version": "1.0.0",
        "session_status": "invalid_status",  # invalid on purpose
        "session_message": "Session created successfully",
        "pipeline_ttl": 3600,
        "session_active": True
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**response_data)


def test_requires_all_fields():
    incomplete_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "session_status": PipelineStatus.FAILED,
        # missing required fields
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**incomplete_data)


def test_validates_field_types():
    response_data = {
        "session_uuid": 123,  # should be str
        "session_endpoint": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "pipeline_version": "1.0.0",
        "session_status": PipelineStatus.RUNNING,
        "session_message": "Session created successfully",
        "pipeline_ttl": 3600,
        "session_active": True
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**response_data)


def test_serializes_to_dict():
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "pipeline_uuid": "pipeline-123",
        "pipeline_version": "1.0.0",
        "session_status": PipelineStatus.RUNNING,
        "session_message": "Session created successfully",
        "pipeline_ttl": 3600,
        "session_active": True
    }

    response = ComputeApiSessionResponse(**response_data)
    serialized = response.model_dump()
    # Enum will be serialized as value
    expected = dict(response_data)
    expected["session_status"] = "running"
    assert serialized == expected


def test_parses_from_json_string():
    json_str = '{"session_uuid": "session-456", "session_endpoint": "https://pipeline.example.com", "pipeline_uuid": "pipeline-123", "pipeline_version": "1.0.0", "session_status": "running", "session_message": "Session created successfully", "pipeline_ttl": 3600, "session_active": true}'

    response = ComputeApiSessionResponse.model_validate_json(json_str)

    assert response.session_endpoint == "https://pipeline.example.com"
    assert response.session_status == PipelineStatus.RUNNING


def test_handles_different_url_formats():
    url_formats = [
        "https://simple.example.com",
        "https://complex.example.com/v1/pipelines/123",
        "http://localhost:8080/pipeline",
        "https://api.example.com/compute/v2/sessions/abc-def",
    ]

    for url in url_formats:
        response_data = {
            "session_uuid": "session-456",
            "session_endpoint": url,
            "pipeline_uuid": "pipeline-123",
            "pipeline_version": "1.0.0",
            "session_status": PipelineStatus.RUNNING,
            "session_message": "Session created successfully",
            "pipeline_ttl": 3600,
            "session_active": True
        }

        response = ComputeApiSessionResponse(**response_data)
        assert response.session_endpoint == url


def test_handles_uuid_formats():
    uuid_formats = [
        "simple-uuid",
        "12345678-1234-1234-1234-123456789012",
        "abc-def-ghi",
        "session_123_456",
    ]

    for uuid_val in uuid_formats:
        response_data = {
            "session_uuid": uuid_val,
            "session_endpoint": "https://pipeline.example.com",
            "pipeline_uuid": uuid_val,
            "pipeline_version": "1.0.0",
            "session_status": PipelineStatus.RUNNING,
            "session_message": "Session created successfully",
            "pipeline_ttl": 3600,
            "session_active": True
        }

        response = ComputeApiSessionResponse(**response_data)
        assert response.pipeline_uuid == uuid_val
        assert response.session_uuid == uuid_val
