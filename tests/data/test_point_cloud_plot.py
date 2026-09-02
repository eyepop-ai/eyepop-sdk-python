import base64
import struct

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eyepop.visualize import EyePopPointCloudPlot, labelled_point_clouds  # noqa: E402

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


def _prediction() -> dict:
    return {"objects": [
        {"classLabel": "person", "x": 0, "y": 0, "width": 4, "height": 4, "mask": _mask(),
         "objects": [{"classLabel": "bag", "x": 0, "y": 0, "width": 2, "height": 2,
                      "mask": _mask(2, 2)}]},
        {"classLabel": "person", "x": 8, "y": 0, "width": 4, "height": 4, "mask": _mask()},
    ]}


@pytest.fixture
def axes():
    figure = plt.figure()
    yield figure.add_subplot(projection="3d")
    plt.close(figure)


def test_labels_number_repeated_classes():
    assert [label for label, _ in labelled_point_clouds(_prediction())] == ["person", "bag", "person 2"]


def test_labels_are_empty_without_clouds():
    assert labelled_point_clouds({"objects": [{"classLabel": "car"}]}) == []
    assert labelled_point_clouds({}) == []


def test_prediction_plots_every_placed_point(axes):
    plot = EyePopPointCloudPlot(axes)
    # 4x4 and 2x2 masks with the diagonal unplaced: 12 + 2 + 12
    assert plot.prediction(_prediction()) == 26


def test_the_point_budget_is_shared_across_clouds(axes):
    # shared rather than per cloud, so adding objects thins the scatter instead
    # of multiplying it
    plot = EyePopPointCloudPlot(axes)
    drawn = plot.prediction(_prediction(), max_points=10)
    assert 0 < drawn <= 13


def test_a_prediction_without_clouds_draws_nothing(axes):
    assert EyePopPointCloudPlot(axes).prediction({"objects": [{"classLabel": "car"}]}) == 0


def test_finish_labels_the_axes_in_metres(axes):
    plot = EyePopPointCloudPlot(axes)
    plot.prediction(_prediction())
    plot.finish(title="clouds")

    assert axes.get_xlabel() == "X (m)"
    assert axes.get_ylabel() == "Y (m)"
    assert axes.get_zlabel() == "Z (m)"
    assert axes.get_title() == "clouds"


def test_finish_survives_an_axis_with_no_extent(axes):
    # every point shares one Z here, and a zero-thick box is rejected by matplotlib
    plot = EyePopPointCloudPlot(axes)
    plot.prediction(_prediction())
    plot.finish()


def test_finish_without_any_points_does_not_raise(axes):
    EyePopPointCloudPlot(axes).finish()
