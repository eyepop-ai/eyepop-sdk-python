import pytest
from pydantic import ValidationError

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext, PipelineStatus


def test_creates_valid_session_response():
    """It creates a valid session response with all required fields."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "pipelines": [{"pipeline_id": "pipeline-123"}],
        "pipeline_uuid": "pipeline-123",
        "pipeline_version": "1.0.0",
        "session_status": PipelineStatus.RUNNING,
        "session_message": "Session created successfully",
        "pipeline_ttl": 3600,
        "session_active": True
    }

    response = ComputeApiSessionResponse(**response_data)

    assert response.session_endpoint == "https://pipeline.example.com"
    assert response.access_token == "jwt-token-123"
    assert response.pipelines == [{"pipeline_id": "pipeline-123"}]
    assert response.pipeline_uuid == "pipeline-123"
    assert response.session_uuid == "session-456"
    assert response.session_status == PipelineStatus.RUNNING
    assert response.session_active is True


def test_session_response_with_defaults():
    """It creates a session response with minimal required fields."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123"
    }

    response = ComputeApiSessionResponse(**response_data)

    assert response.session_endpoint == "https://pipeline.example.com"
    assert response.access_token == "jwt-token-123"
    assert response.pipelines == []
    assert response.pipeline_uuid == ""
    assert response.pipeline_version == ""
    assert response.session_status == PipelineStatus.PENDING
    assert response.session_message == ""
    assert response.pipeline_ttl is None
    assert response.session_active is False


def test_validates_pipeline_status_enum():
    """It only allows valid pipeline status enums."""
    valid_statuses = [
        PipelineStatus.UNKNOWN, 
        PipelineStatus.PENDING, 
        PipelineStatus.RUNNING, 
        PipelineStatus.STOPPED, 
        PipelineStatus.FAILED
    ]

    for status in valid_statuses:
        response_data = {
            "session_uuid": "session-456",
            "session_endpoint": "https://pipeline.example.com",
            "access_token": "jwt-token-123",
            "session_status": status
        }
        response = ComputeApiSessionResponse(**response_data)
        assert response.session_status == status


def test_rejects_invalid_pipeline_status():
    """It rejects invalid pipeline status values."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "session_status": "invalid_status"
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**response_data)


def test_requires_core_fields():
    """It requires session_uuid, session_endpoint, and access_token."""
    incomplete_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com"
        # missing access_token
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**incomplete_data)


def test_validates_field_types():
    """It validates that fields have correct types."""
    response_data = {
        "session_uuid": 123,  # should be str
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123"
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse(**response_data)


def test_serializes_to_dict():
    """It correctly serializes to dictionary."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "pipelines": [{"pipeline_id": "p1"}],
        "session_status": PipelineStatus.RUNNING
    }

    response = ComputeApiSessionResponse(**response_data)
    serialized = response.model_dump()
    
    assert serialized["session_uuid"] == "session-456"
    assert serialized["session_endpoint"] == "https://pipeline.example.com"
    assert serialized["access_token"] == "jwt-token-123"
    assert serialized["pipelines"] == [{"pipeline_id": "p1"}]
    assert serialized["session_status"] == "running"


def test_parses_from_json_string():
    """It parses from JSON string."""
    json_str = '''{
        "session_uuid": "session-456", 
        "session_endpoint": "https://pipeline.example.com", 
        "access_token": "jwt-token-123",
        "pipelines": [],
        "session_status": "running"
    }'''

    response = ComputeApiSessionResponse.model_validate_json(json_str)

    assert response.session_endpoint == "https://pipeline.example.com"
    assert response.access_token == "jwt-token-123"
    assert response.session_status == PipelineStatus.RUNNING


def test_compute_context_creation():
    """It creates a valid ComputeContext."""
    context = ComputeContext(
        compute_url="https://compute.example.com",
        secret_key="test-key",
        access_token="jwt-123",
        session_endpoint="https://session.example.com",
        pipeline_id="pipeline-456"
    )

    assert context.compute_url == "https://compute.example.com"
    assert context.secret_key == "test-key"
    assert context.access_token == "jwt-123"
    assert context.session_endpoint == "https://session.example.com"
    assert context.pipeline_id == "pipeline-456"
    assert context.wait_for_session_timeout == 10
    assert context.wait_for_session_interval == 1


def test_compute_context_defaults():
    """It creates ComputeContext with defaults."""
    context = ComputeContext()

    assert context.compute_url == "https://compute-api.staging.eyepop.xyz"
    assert context.session_endpoint == ""
    assert context.session_uuid == ""
    assert context.pipeline_uuid == ""
    assert context.access_token == ""
    assert context.wait_for_session_timeout == 10
    assert context.wait_for_session_interval == 1


def test_handles_empty_pipelines_list():
    """It handles empty pipelines list."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "pipelines": []
    }

    response = ComputeApiSessionResponse(**response_data)
    assert response.pipelines == []
    assert response.pipeline_uuid == ""


def test_handles_multiple_pipelines():
    """It handles multiple pipelines in the list."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "pipelines": [
            {"pipeline_id": "p1", "name": "Pipeline 1"},
            {"pipeline_id": "p2", "name": "Pipeline 2"}
        ]
    }

    response = ComputeApiSessionResponse(**response_data)
    assert len(response.pipelines) == 2
    assert response.pipelines[0]["pipeline_id"] == "p1"
    assert response.pipelines[1]["pipeline_id"] == "p2"