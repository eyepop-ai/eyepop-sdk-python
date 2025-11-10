import os

import pytest
from pydantic import ValidationError

from eyepop.compute import ComputeApiSessionResponse, ComputeContext
from eyepop.compute.context import PipelineStatus


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

    response = ComputeApiSessionResponse.model_validate(response_data)

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


def test_handles_unknown_pipeline_status_gracefully():
    """It handles unknown pipeline status values using the _missing_ method."""
    # Test the explicitly defined status first
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "session_status": "pipeline_creating"
    }
    response = ComputeApiSessionResponse.model_validate(response_data)
    assert response.session_status == PipelineStatus.PIPELINE_CREATING
    
    # Test various unknown status strings that should be mapped via _missing_
    test_cases = [
        ("pipeline_running", PipelineStatus.RUNNING),   # contains "running"
        ("is_running", PipelineStatus.RUNNING),         # contains "running"
        ("started", PipelineStatus.PENDING),            # contains "start"
        ("creating_pipeline", PipelineStatus.PENDING),  # contains "creat"
        ("failed_to_start", PipelineStatus.FAILED),     # contains "fail"
        ("error_occurred", PipelineStatus.ERROR),       # contains "error"
        ("stopped_by_user", PipelineStatus.STOPPED),    # contains "stop"
        ("completely_unknown", PipelineStatus.UNKNOWN), # no match -> UNKNOWN
        ("random_status", PipelineStatus.UNKNOWN),      # no match -> UNKNOWN
    ]
    
    for status_str, expected_enum in test_cases:
        response_data = {
            "session_uuid": "session-456",
            "session_endpoint": "https://pipeline.example.com",
            "access_token": "jwt-token-123",
            "session_status": status_str
        }
        response = ComputeApiSessionResponse.model_validate(response_data)
        assert response.session_status == expected_enum, f"Status '{status_str}' should map to {expected_enum}"


def test_requires_core_fields():
    """It requires session_uuid, session_endpoint, and access_token."""
    incomplete_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com"
        # missing access_token
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse.model_validate(incomplete_data)


def test_validates_field_types():
    """It validates that fields have correct types."""
    response_data = {
        "session_uuid": 123,  # should be str
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123"
    }

    with pytest.raises(ValidationError):
        ComputeApiSessionResponse.model_validate(response_data)


def test_serializes_to_dict():
    """It correctly serializes to dictionary."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "pipelines": [{"pipeline_id": "p1"}],
        "session_status": PipelineStatus.RUNNING
    }

    response = ComputeApiSessionResponse.model_validate(response_data)
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
        api_key="test-key",
        m2m_access_token="jwt-123",
        session_endpoint="https://session.example.com",
        pipeline_id="pipeline-456"
    )

    assert context.compute_url == "https://compute.example.com"
    assert context.api_key == "test-key"
    assert context.m2m_access_token == "jwt-123"
    assert context.session_endpoint == "https://session.example.com"
    assert context.pipeline_id == "pipeline-456"
    assert context.wait_for_session_timeout == 60
    assert context.wait_for_session_interval == 2


def test_compute_context_defaults():
    """It creates ComputeContext with defaults."""
    os.environ["EYEPOP_URL"] = "https://compute.eyepop.ai"
    context = ComputeContext()

    assert context.compute_url == "https://compute.eyepop.ai"
    assert context.session_endpoint == ""
    assert context.session_uuid == ""
    assert context.pipeline_uuid == ""
    assert context.m2m_access_token == ""
    assert context.wait_for_session_timeout == 60
    assert context.wait_for_session_interval == 2

    del os.environ["EYEPOP_URL"]


def test_handles_empty_pipelines_list():
    """It handles empty pipelines list."""
    response_data = {
        "session_uuid": "session-456",
        "session_endpoint": "https://pipeline.example.com",
        "access_token": "jwt-token-123",
        "pipelines": []
    }

    response = ComputeApiSessionResponse.model_validate(response_data)
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

    response = ComputeApiSessionResponse.model_validate(response_data)
    assert len(response.pipelines) == 2
    assert response.pipelines[0]["pipeline_id"] == "p1"
    assert response.pipelines[1]["pipeline_id"] == "p2"