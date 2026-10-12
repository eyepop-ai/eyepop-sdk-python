import pytest
from pydantic import ValidationError

from eyepop.worker.worker_types import InferenceComponent, Pop


def test_inference_component_sends_classes():
    component = InferenceComponent(ability="eyepop.common-objects:latest", classes=["person", "car"])

    assert component.model_dump(exclude_none=True)["classes"] == ["person", "car"]


def test_pop_reads_classes():
    pop = Pop.model_validate(
        {"components": [{"type": "inference", "ability": "eyepop.common-objects:latest", "classes": ["person"]}]}
    )

    assert pop.components[0].classes == ["person"]


def test_classes_must_be_a_list_of_labels():
    with pytest.raises(ValidationError):
        InferenceComponent(ability="eyepop.common-objects:latest", classes="person")
