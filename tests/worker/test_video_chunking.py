"""A Pop component's chunking and a source's own, by the same names (VREE-352)."""

import math

import pytest
from pydantic import ValidationError

from eyepop.worker.worker_types import ComponentParams, InferenceComponent


def test_a_component_takes_a_chunk_length_above_zero_and_an_overlap_below_half():
    component = InferenceComponent(ability='eyepop.qwen3.event-classify:latest', videoChunkLengthSeconds=4,
                                   videoChunkOverlap=0.25)
    assert component.videoChunkLengthSeconds == 4
    assert component.videoChunkOverlap == 0.25
    assert InferenceComponent(ability='a', videoChunkOverlap=-1).videoChunkOverlap == -1  # gaps


@pytest.mark.parametrize('field,value', [
    ('videoChunkLengthSeconds', 0),
    ('videoChunkLengthSeconds', -2),
    ('videoChunkLengthSeconds', math.inf),
    ('videoChunkOverlap', 0.5),
    ('videoChunkOverlap', 0.75),
    ('videoChunkOverlap', math.nan),
])
def test_a_component_refuses_an_invalid_chunking(field, value):
    with pytest.raises(ValidationError):
        InferenceComponent(ability='a', **{field: value})


def test_a_sources_params_set_its_own_chunking_by_the_same_names():
    params = ComponentParams(componentId=2, values={'videoChunkLengthSeconds': 4, 'videoChunkOverlap': -0.5})
    assert params.values['videoChunkLengthSeconds'] == 4
    assert ComponentParams(componentId=2, values={'prompts': [], 'videoChunkOverlap': None}).componentId == 2


@pytest.mark.parametrize('values', [
    {'videoChunkLengthSeconds': 0},
    {'videoChunkLengthSeconds': '4'},
    {'videoChunkLengthSeconds': True},
    {'videoChunkOverlap': 0.5},
    {'videoChunkOverlap': math.inf},
])
def test_a_sources_params_refuse_an_invalid_chunking(values):
    with pytest.raises(ValidationError, match='videoChunk'):
        ComponentParams(componentId=2, values=values)
