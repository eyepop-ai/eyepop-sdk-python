from typing import List

from pydantic import BaseModel


class Point2d(BaseModel):
    x: float
    y: float


class Point3d(Point2d):
    z: float | None = None


class Box(BaseModel):
    topLeft: Point2d
    bottomRight: Point2d


class Contour(BaseModel):
    points: List[Point2d]
    cutouts: List[List[Point2d]]


class Mask(BaseModel):
    bitmap: str
    width: int
    height: int
    stride: int


class Depth(BaseModel):
    """A frame-level depth map as produced by depth estimation abilities (e.g. eyepop.depth.*).

    `values` is the base64 encoding of width*height little-endian float32 values in
    row-major order. The map has the aspect ratio of the source frame; map a source
    coordinate proportionally: (x * width / source_width, y * height / source_height).
    Values are canonical metric depth (multiply by the camera's focal length in pixels
    and divide by 300 for meters). Sky pixels carry +Infinity.
    Use eyepop.DepthMap to decode and sample the values.
    """

    width: int
    height: int
    values: str
