from typing import List

from pydantic import BaseModel


class Point2d(BaseModel):
    """A point in source pixels, optionally placed in 3D.

    `worldX`/`worldY`/`worldZ` are metres, present only when the pipeline was
    asked for world coordinates (a Pop with `depthMapAbility` and a component
    with `translateToWorld`) and only in prediction v2. A point the worker could
    not place - sky, outside the depth map, no usable map - carries none of the
    three rather than a zero or a NaN, so test for None.

    Which frame they are in depends on the source's camera: with extrinsics, the
    world frame those define (Z up, ground at Z = 0); without them, the camera
    frame in the OpenCV convention (X right, Y down, Z forward from the camera).

    Only the carriers the worker enriches ever populate them: key points,
    outline points and contour points. Bounding boxes and mesh points do not -
    a box is not a point, and any single anchor choice would be arbitrary.
    """

    x: float
    y: float
    worldX: float | None = None
    worldY: float | None = None
    worldZ: float | None = None


class Point3d(Point2d):
    z: float | None = None


class Box(BaseModel):
    topLeft: Point2d
    bottomRight: Point2d


class Contour(BaseModel):
    points: List[Point2d]
    cutouts: List[List[Point2d]]


class Mask(BaseModel):
    """A segmentation mask, optionally with a per-object point cloud.

    `world` is the base64 encoding of three little-endian float32 values per
    mask pixel, row-major, exactly width*height triples - so the point for
    bitmap pixel (i, j) is at triple index j * width + i. Points the worker
    could not place are NaN. Use eyepop.PointCloud to decode and sample it.
    """

    bitmap: str
    width: int
    height: int
    stride: int
    world: str | None = None


class Depth(BaseModel):
    """A frame-level depth map as produced by depth estimation abilities (e.g. eyepop.depth.*).

    `values` is the base64 encoding of width*height little-endian float32 values in
    row-major order. The map has the aspect ratio of the source frame; map a source
    coordinate proportionally: (x * width / source_width, y * height / source_height).
    Sky pixels carry +Infinity. Use eyepop.DepthMap to decode and sample the values.

    `semantic` says what the values mean, and is always present in prediction v2 -
    "unknown" included - so an absent member means a worker that predates the field
    rather than a map that declined to say:

    * "canonical_metric" - metres = value * focal_px / 300, with focal_px scaled to
      the map's own resolution
    * "metric" - the value is already metres
    * "relative" - scale- AND shift-invariant, so ordering is meaningful but distance
      is not. Not back-projectable: recovering a cloud from it yields a distorted
      scene rather than a scaled one
    * "unknown" - the ability declared nothing. Not back-projectable
    """

    width: int
    height: int
    values: str
    semantic: str | None = None
