import base64
import struct

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eyepop.visualize import EyePopWorldPlot, labelled_world_points  # noqa: E402

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
    labels = [label for label, _ in labelled_world_points(_prediction())]
    assert labels == [
        "person keypoints", "person outline", "person contour", "person cutout", "person mask",
        "bag mask", "person 2 mask",
    ]


def test_a_pop_without_masks_still_has_something_to_show():
    prediction = {"objects": [{"classLabel": "person", "keyPoints": [{"points": _placed(1.0)}]}]}
    labels = [label for label, _ in labelled_world_points(prediction)]
    assert labels == ["person keypoints"]


def test_frame_level_key_points_are_collected():
    # abilities that produce key points without an enclosing object
    labelled = labelled_world_points({"keyPoints": [{"points": _placed(1.0, 2.0)}]})
    assert [label for label, _ in labelled] == ["keypoints"]
    assert len(labelled[0][1]) == 2


def test_points_without_world_members_are_left_out():
    prediction = {"objects": [{"classLabel": "person", "outline": [{"x": 1.0, "y": 2.0}]}]}
    assert labelled_world_points(prediction) == []
    assert labelled_world_points({"objects": [{"classLabel": "car"}]}) == []
    assert labelled_world_points({}) == []


def test_collected_series_are_metre_triples():
    labelled = dict(labelled_world_points(_prediction()))
    assert labelled["person outline"].shape == (1, 3)
    assert labelled["person keypoints"].shape == (2, 3)  # the unplaced third is dropped
    assert labelled["person outline"].tolist() == [[1.0, 2.0, 3.0]]


def test_prediction_plots_every_placed_point(axes):
    plot = EyePopWorldPlot(axes)
    # 12 + 2 + 12 mask points, plus 2 key points, outline, contour and cutout
    assert plot.prediction(_prediction()) == 31


def test_sparse_series_survive_a_tight_budget(axes):
    # a skeleton and a dense cloud are both series; a budget shared evenly would
    # thin the skeleton away to save a fraction of the cloud
    plot = EyePopWorldPlot(axes)
    drawn = plot.prediction(_prediction(), max_points=1)

    sparse = sum(len(points) for label, points in labelled_world_points(_prediction())
                 if "mask" not in label)
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

    spans = [round(high - low) for low, high in
             (axes.get_xlim(), axes.get_ylim(), axes.get_zlim())]
    assert spans == [1, 3, 2]


def test_finish_survives_an_axis_with_no_extent(axes):
    # a zero-thick box is rejected by matplotlib
    plot = EyePopWorldPlot(axes)
    plot.points([np.array([[1.0, 2.0, 3.0]], dtype="float32")])
    plot.finish()


def test_finish_without_any_points_does_not_raise(axes):
    EyePopWorldPlot(axes).finish()
