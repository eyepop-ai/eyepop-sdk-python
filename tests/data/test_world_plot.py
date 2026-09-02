import base64
import struct

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

from eyepop.visualize import (  # noqa: E402
    POSE_2D_CONNECTIONS,
    EyePopWorldPlot,
    labelled_world_points,
)

_NAN = float("nan")


def _cloud_b64(width: int, height: int) -> str:
    values = []
    for j in range(height):
        for i in range(width):
            values.extend((_NAN, _NAN, _NAN) if i == j else (float(i), float(j), 2.0))
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode()


def _mask(width: int = 4, height: int = 4) -> dict:
    return {"bitmap": "", "width": width, "height": height, "stride": width,
            "world": _cloud_b64(width, height)}


def _placed(*zs: float) -> list[dict]:
    return [{"x": 0.0, "y": 0.0, "worldX": 1.0, "worldY": 2.0, "worldZ": z} for z in zs]


def _joint(label: str, z: float) -> dict:
    return {"x": 0.0, "y": 0.0, "classLabel": label, "worldX": 1.0, "worldY": 2.0, "worldZ": z}


def _pose(*labels: str, category: str = "2d-body-points") -> dict:
    return {"category": category,
            "points": [_joint(label, float(index)) for index, label in enumerate(labels)]}


def _prediction() -> dict:
    return {"objects": [
        {"classLabel": "person", "x": 0, "y": 0, "width": 4, "height": 4, "mask": _mask(),
         "keyPoints": [{"points": _placed(1.0, 2.0) + [{"x": 0.0, "y": 0.0}]}],
         "outline": _placed(3.0),
         "contours": [{"points": _placed(4.0), "cutouts": [_placed(5.0)]}],
         "objects": [{"classLabel": "bag", "x": 0, "y": 0, "width": 2, "height": 2,
                      "mask": _mask(2, 2)}]},
        {"classLabel": "person", "x": 8, "y": 0, "width": 4, "height": 4, "mask": _mask()},
    ]}


@pytest.fixture
def axes():
    figure = plt.figure()
    yield figure.add_subplot(projection="3d")
    plt.close(figure)


def test_every_carrier_is_collected():
    # key points, outline, contour, cutout and mask alike, not only the clouds
    labels = [entry.label for entry in labelled_world_points(_prediction())]
    assert labels == [
        "person keypoints", "person outline", "person contour", "person cutout", "person mask",
        "bag mask", "person 2 mask",
    ]


def test_a_pop_without_masks_still_has_something_to_show():
    prediction = {"objects": [{"classLabel": "person", "keyPoints": [{"points": _placed(1.0)}]}]}
    labels = [entry.label for entry in labelled_world_points(prediction)]
    assert labels == ["person keypoints"]


def test_frame_level_key_points_are_collected():
    # abilities that produce key points without an enclosing object
    labelled = labelled_world_points({"keyPoints": [{"points": _placed(1.0, 2.0)}]})
    assert [entry.label for entry in labelled] == ["keypoints"]
    assert len(labelled[0].points) == 2


def test_points_without_world_members_are_left_out():
    prediction = {"objects": [{"classLabel": "person", "outline": [{"x": 1.0, "y": 2.0}]}]}
    assert labelled_world_points(prediction) == []
    assert labelled_world_points({"objects": [{"classLabel": "car"}]}) == []
    assert labelled_world_points({}) == []


def test_collected_series_are_metre_triples():
    labelled = {entry.label: entry for entry in labelled_world_points(_prediction())}
    assert labelled["person outline"].points.shape == (1, 3)
    # the unplaced third key point is dropped from the drawn points
    assert labelled["person keypoints"].points.shape == (2, 3)
    assert labelled["person outline"].points.tolist() == [[1.0, 2.0, 3.0]]


def test_prediction_plots_every_placed_point(axes):
    plot = EyePopWorldPlot(axes)
    # 12 + 2 + 12 mask points, plus 2 key points, outline, contour and cutout
    assert plot.prediction(_prediction()) == 31


def test_sparse_series_survive_a_tight_budget(axes):
    # a skeleton and a dense cloud are both series; a budget shared evenly would
    # thin the skeleton away to save a fraction of the cloud
    plot = EyePopWorldPlot(axes)
    drawn = plot.prediction(_prediction(), max_points=1)

    sparse = sum(len(entry.points) for entry in labelled_world_points(_prediction())
                 if "mask" not in entry.label)
    assert sparse == 5
    assert drawn >= sparse  # every sparse series drawn whole despite the budget


def test_a_prediction_without_world_coordinates_draws_nothing(axes):
    assert EyePopWorldPlot(axes).prediction({"objects": [{"classLabel": "car"}]}) == 0


def test_finish_labels_the_axes_in_metres(axes):
    plot = EyePopWorldPlot(axes)
    plot.prediction(_prediction())
    plot.finish(title="world")

    # Z into the scene and Y up the page, not component order
    assert axes.get_xlabel() == "X (m)"
    assert axes.get_ylabel() == "Z (m)"
    assert axes.get_zlabel() == "Y (m)"
    assert axes.get_title() == "world"


def test_the_drawn_axes_swap_y_and_z(axes):
    # X spans 1, Y spans 2 and Z spans 3, so whichever screen axis spans 3 is
    # the one Z was drawn against
    plot = EyePopWorldPlot(axes)
    plot.points([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype="float32")])
    plot.finish()

    # magnitudes, since the vertical axis is deliberately inverted
    spans = [round(abs(high - low)) for low, high in
             (axes.get_xlim(), axes.get_ylim(), axes.get_zlim())]
    assert spans == [1, 3, 2]


def test_finish_survives_an_axis_with_no_extent(axes):
    # a zero-thick box is rejected by matplotlib
    plot = EyePopWorldPlot(axes)
    plot.points([np.array([[1.0, 2.0, 3.0]], dtype="float32")])
    plot.finish()


def test_finish_without_any_points_does_not_raise(axes):
    EyePopWorldPlot(axes).finish()


def test_the_vertical_axis_is_inverted_so_a_camera_frame_scene_stands_upright(axes):
    # camera frame Y grows downwards, so drawn as-is a figure's feet sit above
    # its head; the axis is flipped rather than the coordinates
    plot = EyePopWorldPlot(axes)
    plot.points([np.array([[0.0, 0.2, 3.0], [0.0, 1.8, 3.0]], dtype="float32")])
    plot.finish()

    low, high = axes.get_zlim()
    assert low > high


def test_inversion_is_idempotent(axes):
    plot = EyePopWorldPlot(axes)
    plot.points([np.array([[0.0, 0.2, 3.0], [0.0, 1.8, 3.0]], dtype="float32")])
    plot.finish()
    first = axes.get_zlim()
    plot.finish()

    assert axes.get_zlim() == first


def test_a_z_up_world_frame_keeps_the_axis_as_it_is(axes):
    plot = EyePopWorldPlot(axes, invert_y=False)
    plot.points([np.array([[0.0, 0.2, 3.0], [0.0, 1.8, 3.0]], dtype="float32")])
    plot.finish()

    low, high = axes.get_zlim()
    assert low < high


def _series(prediction: dict) -> dict:
    return {entry.label: entry for entry in labelled_world_points(prediction)}


def test_contour_points_are_connected_in_order_and_closed():
    prediction = {"objects": [{"classLabel": "leaf",
                               "contours": [{"points": _placed(1.0, 2.0, 3.0), "cutouts": []}]}]}
    contour = _series(prediction)["leaf contour"]

    # three points, closed, so three segments
    assert contour.segments.shape == (3, 2, 3)
    assert contour.segments[0].tolist() == [[1.0, 2.0, 1.0], [1.0, 2.0, 2.0]]
    assert contour.segments[2].tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 1.0]]  # the closing edge


def test_an_unplaced_point_breaks_a_contour_rather_than_being_bridged():
    # dropping the hole first would join its neighbours across a gap that is
    # not there
    points = _placed(1.0) + [{"x": 0.0, "y": 0.0}] + _placed(3.0)
    prediction = {"objects": [{"classLabel": "leaf",
                               "contours": [{"points": points, "cutouts": []}]}]}
    contour = _series(prediction)["leaf contour"]

    assert len(contour.points) == 2       # both placed points are drawn
    assert contour.segments.shape == (1, 2, 3)  # only the closing edge survives
    assert contour.segments[0].tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 1.0]]


def test_outlines_and_cutouts_are_connected_too():
    prediction = {"objects": [{"classLabel": "leaf",
                               "outline": _placed(1.0, 2.0, 3.0),
                               "contours": [{"points": _placed(1.0, 2.0, 3.0),
                                             "cutouts": [_placed(4.0, 5.0, 6.0)]}]}]}
    series = _series(prediction)
    assert series["leaf outline"].segments.shape == (3, 2, 3)
    assert series["leaf cutout"].segments.shape == (3, 2, 3)


def test_key_points_are_connected_by_the_pose_skeleton():
    prediction = {"objects": [{"classLabel": "person",
                               "keyPoints": [_pose("left shoulder", "right shoulder",
                                                   "left elbow", "left wrist")]}]}
    keypoints = _series(prediction)["person keypoints"]

    # left-right shoulder, shoulder-elbow, elbow-wrist; the other connections
    # in the table have no points here
    assert keypoints.segments.shape == (3, 2, 3)


def test_the_skeleton_matches_by_label_not_by_index():
    # the same joints in a different order must give the same skeleton
    forwards = _series({"objects": [{"classLabel": "person",
                                     "keyPoints": [_pose("left shoulder", "left elbow")]}]})
    backwards = _series({"objects": [{"classLabel": "person",
                                      "keyPoints": [_pose("left elbow", "left shoulder")]}]})

    assert forwards["person keypoints"].segments.shape == (1, 2, 3)
    assert backwards["person keypoints"].segments.shape == (1, 2, 3)


def test_a_joint_the_worker_could_not_place_drops_its_bones():
    pose = _pose("left shoulder", "left elbow", "left wrist")
    del pose["points"][1]["worldZ"]  # the elbow was not placed
    keypoints = _series({"objects": [{"classLabel": "person", "keyPoints": [pose]}]})["person keypoints"]

    assert len(keypoints.points) == 2
    assert keypoints.segments.shape == (0, 2, 3)  # both bones needed the elbow


def test_an_unknown_key_point_category_gets_no_lines():
    # points joined in whatever order they arrived would be meaningless
    pose = _pose("a", "b", "c", category="something-else")
    keypoints = _series({"objects": [{"classLabel": "x", "keyPoints": [pose]}]})["x keypoints"]

    assert len(keypoints.points) == 3
    assert keypoints.segments.size == 0


def test_the_3d_pose_skeleton_is_recognised():
    pose = _pose("left shoulder", "left elbow", "left thumb", "left wrist",
                 category="3d-body-points")
    keypoints = _series({"objects": [{"classLabel": "person", "keyPoints": [pose]}]})["person keypoints"]

    # shoulder-elbow, elbow-wrist, wrist-thumb
    assert keypoints.segments.shape == (3, 2, 3)


def test_a_mask_cloud_has_no_connections():
    # a grid rather than a path, so there is nothing to connect
    assert _series(_prediction())["person mask"].segments.size == 0


def test_connections_are_drawn_on_the_axes(axes):
    prediction = {"objects": [{"classLabel": "person",
                               "keyPoints": [_pose("left shoulder", "right shoulder")]}]}
    plot = EyePopWorldPlot(axes)
    plot.prediction(prediction)

    assert any(isinstance(collection, Line3DCollection) for collection in axes.collections)


def test_the_skeleton_table_matches_the_2d_renderer():
    # mirrored from the Node SDK's render-pose.ts; these pairs are the contract
    assert ("left shoulder", "right shoulder") in POSE_2D_CONNECTIONS
    assert ("left knee", "left ankle") in POSE_2D_CONNECTIONS
    assert len(POSE_2D_CONNECTIONS) == 12
