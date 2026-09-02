import base64
import math
import struct

import numpy as np
import pytest

from eyepop import PointCloud
from eyepop.data.types import Contour, Depth, Mask, Point2d, Prediction
from eyepop.data.types.prediction import PredictedKeyPoint, PredictedKeyPoints, PredictedObject

# A 2x2 mask cloud: three placed points and one the worker could not place,
# which the wire marks with NaN rather than omitting.
_NAN = float("nan")
_CLOUD = [
    1.0, 2.0, 3.0,      # (0, 0)
    4.0, 5.0, 6.0,      # (1, 0)
    _NAN, _NAN, _NAN,   # (0, 1) - sky, out of map, or not covered by the mask
    7.0, 8.0, 9.0,      # (1, 1)
]
_CLOUD_B64 = base64.b64encode(struct.pack("<12f", *_CLOUD)).decode()


def _mask_dict() -> dict:
    return {"bitmap": "", "width": 2, "height": 2, "stride": 2, "world": _CLOUD_B64}


def test_point_carries_world_coordinates():
    point = Point2d(x=10.0, y=20.0, worldX=1.5, worldY=-2.25, worldZ=8.0)
    assert point.worldZ == 8.0


def test_point_without_world_coordinates_leaves_them_none():
    point = Point2d(x=10.0, y=20.0)
    assert point.worldX is None
    assert point.worldY is None
    assert point.worldZ is None


def test_key_point_z_is_independent_of_world_z():
    point = PredictedKeyPoint(x=1.0, y=2.0, z=0.5, worldZ=4.125)
    assert point.z == 0.5
    assert point.worldZ == 4.125


def test_outline_and_contour_points_deserialise_world_coordinates():
    obj = PredictedObject(**{
        "classLabel": "person", "x": 0, "y": 0, "width": 10, "height": 10,
        "outline": [{"x": 1, "y": 2, "worldX": 0.1, "worldY": 0.2, "worldZ": 0.3}],
        "contours": [{
            "points": [{"x": 3, "y": 4, "worldX": 1.0, "worldY": 1.1, "worldZ": 1.2}],
            "cutouts": [[{"x": 5, "y": 6, "worldX": 2.0, "worldY": 2.1, "worldZ": 2.2}]],
        }],
    })
    assert obj.outline is not None and obj.outline[0].worldZ == 0.3
    assert obj.contours is not None
    assert obj.contours[0].points[0].worldX == 1.0
    assert obj.contours[0].cutouts[0][0].worldZ == 2.2


def test_prediction_without_world_members_is_unchanged():
    prediction = Prediction(**{
        "source_width": 1280, "source_height": 720,
        "objects": [{"classLabel": "person", "x": 0, "y": 0, "width": 10, "height": 10,
                     "outline": [{"x": 1, "y": 2}]}],
        "keyPoints": [{"points": [{"x": 3, "y": 4, "z": 0.5}]}],
    })
    assert prediction.objects is not None
    outline = prediction.objects[0].outline
    assert outline is not None and outline[0].worldX is None
    assert prediction.keyPoints is not None
    assert prediction.keyPoints[0].points[0].z == 0.5
    assert prediction.keyPoints[0].points[0].worldZ is None


def test_a_point_the_worker_could_not_place_carries_no_world_members():
    # the wire omits all three rather than emitting a zero or a NaN, so the
    # distinction a caller needs is None, not a sentinel value
    points = PredictedKeyPoints(points=[
        PredictedKeyPoint(x=1.0, y=1.0, worldX=1.0, worldY=2.0, worldZ=3.0),
        PredictedKeyPoint(x=2.0, y=2.0),
    ])
    assert [p.worldZ is None for p in points.points] == [False, True]


def test_world_coordinates_round_trip_through_model_dump():
    contour = Contour(points=[Point2d(x=1, y=2, worldX=3, worldY=4, worldZ=5)], cutouts=[])
    assert Contour(**contour.model_dump()) == contour


def test_depth_carries_semantic():
    depth = Depth(width=2, height=1, values="", semantic="canonical_metric")
    assert depth.semantic == "canonical_metric"


def test_absent_depth_semantic_is_distinguishable_from_unknown():
    # always present in v2, "unknown" included, so absent means a worker that
    # predates the field rather than a map that declined to say
    assert Depth(width=2, height=1, values="").semantic is None
    assert Depth(width=2, height=1, values="", semantic="unknown").semantic == "unknown"


def test_mask_without_a_cloud_has_no_world():
    assert Mask(bitmap="", width=2, height=2, stride=2).world is None


def test_point_cloud_decodes_to_a_height_width_3_array():
    cloud = PointCloud.from_mask(_mask_dict())
    assert cloud is not None
    assert cloud.array.shape == (2, 2, 3)
    assert cloud.array.dtype == np.dtype("float32")


def test_point_cloud_indexes_like_the_bitmap():
    cloud = PointCloud.from_mask(_mask_dict())
    assert cloud is not None
    # the point for bitmap pixel (i, j) is at triple index j * width + i
    assert cloud.at(0, 0) == (1.0, 2.0, 3.0)
    assert cloud.at(1, 0) == (4.0, 5.0, 6.0)
    assert cloud.at(1, 1) == (7.0, 8.0, 9.0)


def test_nan_is_preserved_as_the_omission_sentinel():
    cloud = PointCloud.from_mask(_mask_dict())
    assert cloud is not None
    assert cloud.at(0, 1) is None
    assert bool(np.isnan(cloud.array[1, 0]).all())
    assert cloud.placed_mask.tolist() == [[True, True], [False, True]]


def test_the_depth_map_nan_validator_is_not_in_the_point_cloud_path():
    # DepthMap.array rejects any NaN; a cloud uses NaN as its omission sentinel,
    # so sharing that validator would reject every valid cloud
    from eyepop.depth_map import DepthMap

    values = base64.b64encode(struct.pack("<4f", 1.0, _NAN, 2.0, 3.0)).decode()
    with pytest.raises(ValueError):
        _ = DepthMap(2, 2, values).array

    cloud = PointCloud.from_mask(_mask_dict())
    assert cloud is not None
    assert cloud.array.shape == (2, 2, 3)  # no raise


def test_point_cloud_accepts_the_pydantic_mask_model():
    cloud = PointCloud.from_mask(Mask(**_mask_dict()))
    assert cloud is not None
    assert cloud.at(0, 0) == (1.0, 2.0, 3.0)


def test_point_cloud_from_object_maps_source_coordinates():
    obj = PredictedObject(**{
        "classLabel": "person", "x": 100.0, "y": 200.0, "width": 40.0, "height": 80.0,
        "mask": _mask_dict(),
    })
    cloud = PointCloud.from_object(obj)
    assert cloud is not None
    # the mask spans the bounding box, so the left half of it is column 0
    assert cloud.at_source(110.0, 220.0) == (1.0, 2.0, 3.0)
    assert cloud.at_source(130.0, 220.0) == (4.0, 5.0, 6.0)
    assert cloud.at_source(130.0, 260.0) == (7.0, 8.0, 9.0)
    assert cloud.at_source(110.0, 260.0) is None


def test_at_source_needs_the_bounding_box():
    cloud = PointCloud.from_mask(_mask_dict())
    assert cloud is not None
    with pytest.raises(ValueError):
        cloud.at_source(1.0, 1.0)


def test_point_cloud_is_none_without_a_world_member():
    assert PointCloud.from_mask({"bitmap": "", "width": 2, "height": 2, "stride": 2}) is None
    assert PointCloud.from_mask(None) is None
    assert PointCloud.from_object({"x": 0, "y": 0, "width": 1, "height": 1}) is None


def test_point_cloud_rejects_a_payload_of_the_wrong_size():
    short = base64.b64encode(struct.pack("<3f", 1.0, 2.0, 3.0)).decode()
    with pytest.raises(ValueError):
        _ = PointCloud(2, 2, short).array


def test_point_cloud_rejects_indices_outside_the_mask():
    cloud = PointCloud.from_mask(_mask_dict())
    assert cloud is not None
    with pytest.raises(IndexError):
        cloud.at(2, 0)


def test_prediction_with_a_mask_cloud_deserialises():
    prediction = Prediction(**{
        "source_width": 640, "source_height": 480,
        "objects": [{"classLabel": "car", "x": 0, "y": 0, "width": 2, "height": 2,
                     "mask": _mask_dict()}],
        "depth": {"width": 2, "height": 1, "values": "", "semantic": "metric"},
    })
    assert prediction.objects is not None
    mask = prediction.objects[0].mask
    assert mask is not None and mask.world == _CLOUD_B64
    assert prediction.depth is not None and prediction.depth.semantic == "metric"
    assert math.isnan(PointCloud.from_object(prediction.objects[0]).array[1, 0, 0])  # type: ignore[union-attr]
