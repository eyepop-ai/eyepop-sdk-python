"""The area types are part of the public surface of `eyepop.data.types`.

`roi` on a worker job is typed `Area`, and `Roi.area` holds one, so a caller
building or reading either needs to be able to name the type. `ContourArea`,
`AreaType` and the `Area` union were reachable only through
`eyepop.data.types.asset` while `RectangleArea` beside them was re-exported, so
the obvious import worked for one shape and failed for the other.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from eyepop.data.types import Area, AreaType, ContourArea, Point2d, RectangleArea, Roi


def test_every_area_type_is_importable_from_the_package():
    """The import in the docs, and the one a reader would try first."""
    assert AreaType.RECTANGLE == "rectangle"
    assert AreaType.CONTOUR == "contour"


def test_a_rectangle_carries_its_discriminator():
    rectangle = RectangleArea(x=160, y=0, width=320, height=480)
    assert rectangle.type == AreaType.RECTANGLE


def test_a_contour_carries_its_discriminator_and_points():
    contour = ContourArea(points=[Point2d(x=220, y=90), Point2d(x=540, y=110), Point2d(x=610, y=470)])
    assert contour.type == AreaType.CONTOUR
    assert [point.x for point in contour.points] == [220, 540, 610]


def test_a_roi_holds_either_shape():
    """`Roi` was already re-exported while the type of its `area` was not."""
    contour = ContourArea(points=[Point2d(x=0, y=0), Point2d(x=10, y=0), Point2d(x=5, y=8)])
    assert Roi(name="lane", area=contour).area.type == AreaType.CONTOUR
    assert Roi(name="bay", area=RectangleArea(x=1, y=2, width=3, height=4)).area.type == AreaType.RECTANGLE


def test_an_area_is_discriminated_on_the_wire():
    """The union is what a worker job's `roi` is typed as, so it has to resolve
    the right shape from JSON rather than guessing by field."""
    adapter = TypeAdapter(Area)

    contour = adapter.validate_python(
        {"type": "contour", "points": [{"x": 1, "y": 1}, {"x": 5, "y": 1}, {"x": 3, "y": 4}]}
    )
    assert isinstance(contour, ContourArea)

    rectangle = adapter.validate_python({"type": "rectangle", "x": 1, "y": 2, "width": 3, "height": 4})
    assert isinstance(rectangle, RectangleArea)

    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "ellipse"})
