import base64
import math
import struct

import pytest

from eyepop import DepthMap
from eyepop.data.data_top_k import filter_prediction_top_k
from eyepop.data.types import Depth, PredictedClass, Prediction

# 4x2 map, row-major; one +Infinity sky pixel at (3, 0)
_VALUES = [1.5, 2.5, 3.5, math.inf, 10.0, 20.0, 30.0, 40.0]
_VALUES_B64 = base64.b64encode(struct.pack("<8f", *_VALUES)).decode()


def _depth_dict() -> dict:
    return {"width": 4, "height": 2, "values": _VALUES_B64}


def test_prediction_model_carries_depth():
    prediction = Prediction(source_width=1280, source_height=640, depth=Depth(**_depth_dict()))
    assert prediction.depth is not None
    assert prediction.depth.width == 4
    round_tripped = Prediction(**prediction.model_dump())
    assert round_tripped.depth == prediction.depth


def test_from_prediction_decodes_worker_dict():
    depth_map = DepthMap.from_prediction({"source_width": 1280, "depth": _depth_dict()})
    assert depth_map is not None
    assert depth_map.width == 4
    assert depth_map.height == 2
    assert depth_map.array.shape == (2, 4)
    assert list(depth_map.array.flatten()[:3]) == [1.5, 2.5, 3.5]
    assert math.isinf(depth_map.array[0, 3])


def test_from_prediction_accepts_pydantic_model():
    prediction = Prediction(source_width=1280, source_height=640, depth=Depth(**_depth_dict()))
    depth_map = DepthMap.from_prediction(prediction)
    assert depth_map is not None
    assert depth_map.at(0, 0) == 1.5


def test_from_prediction_without_depth_returns_none():
    assert DepthMap.from_prediction({"source_width": 1280}) is None
    assert DepthMap.from_prediction(Prediction(source_width=1.0, source_height=1.0)) is None


def test_sky_and_finite_range():
    depth_map = DepthMap.from_prediction({"depth": _depth_dict()})
    assert depth_map is not None
    assert depth_map.sky_mask[0, 3]
    assert not depth_map.sky_mask[0, 0]
    assert depth_map.finite_min == 1.5
    assert depth_map.finite_max == 40.0
    assert depth_map.is_sky(3, 0)
    assert not depth_map.is_sky(0, 0)


def test_at_maps_source_coordinates_proportionally():
    depth_map = DepthMap.from_prediction({"depth": _depth_dict()})
    assert depth_map is not None
    assert depth_map.at(0, 0, source_width=1280, source_height=640) == 1.5
    assert depth_map.at(1279, 639, source_width=1280, source_height=640) == 40.0
    # out-of-bounds map coordinates clamp
    assert depth_map.at(99, 99) == 40.0


def test_truncated_payload_raises_value_error():
    truncated = base64.b64encode(struct.pack("<2f", 1.0, 2.0)).decode()
    depth_map = DepthMap.from_prediction({"depth": {"width": 4, "height": 2, "values": truncated}})
    assert depth_map is not None
    with pytest.raises(ValueError):
        _ = depth_map.array


def test_top_k_preserves_depth_and_other_members():
    prediction = Prediction(
        source_width=1280,
        source_height=640,
        timestamp=42,
        classes=[PredictedClass(classLabel=f"class_{i}", confidence=i / 10.0) for i in range(5)],
        depth=Depth(**_depth_dict()),
    )
    filtered = filter_prediction_top_k(prediction, 2)
    assert filtered.classes is not None and len(filtered.classes) == 2
    assert filtered.depth == prediction.depth
    assert filtered.timestamp == 42
