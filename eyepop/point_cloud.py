"""Decoding and sampling of point clouds.

When a pipeline is asked for world coordinates, an object carrying a
segmentation mask also carries `mask.world`: three little-endian float32 values
per mask pixel, row-major, exactly `mask.width * mask.height` triples. The point
for bitmap pixel (i, j) sits at triple index `j * mask.width + i`, so the cloud
indexes exactly like the bitmap with no separate lookup.

A Pop asking for `depthMap.toWorld` also gets `depth.world`, the same encoding
on the depth map's own grid: one point per map pixel rather than per mask pixel,
covering the whole scene. `PointCloud.from_depth` reads it.

Points the worker could not place are NaN - sky pixels, samples outside the
depth map, and pixels the mask does not cover. NaN is the omission sentinel
here, which is why this does not share DepthMap's validator: that one rejects
any NaN, and so would reject every valid cloud.
"""

import base64
from typing import Any

import numpy as np

from eyepop.data.types.common import Depth, Mask

# One xyz triple per pixel, so a cloud is three float32 per mask pixel.
_VALUES_PER_POINT = 3
_BYTES_PER_VALUE = 4


class PointCloud:
    """A decoded point cloud, per object or per scene.

    Create with `PointCloud.from_object(obj)`, `PointCloud.from_mask(mask)` or
    `PointCloud.from_depth(depth)`; access the decoded points as a numpy
    float32 array of shape (height, width, 3) via `.array`, or sample one point
    with `.at(i, j)`.

    Coordinates are metres, in the frame the source's camera extrinsics define
    (Z up, ground at Z = 0) or the OpenCV camera frame when it supplied none.

    The grid is whatever the cloud was made from: an object's mask, which spans
    its bounding box rather than the frame, or the depth map itself, which
    spans the whole frame. Either way (i, j) is a source coordinate only via
    that box, which `from_object` and `from_depth` record and `from_mask`
    cannot know - so `.at_source(x, y)` needs one of the first two.
    """

    def __init__(self, width: int, height: int, world: str, box: tuple[float, float, float, float] | None = None):
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid point cloud dimensions: width={width!r} height={height!r}")
        self.width = width
        self.height = height
        self._world = world
        self._box = box
        self._array: np.ndarray | None = None

    @staticmethod
    def from_mask(mask: dict[str, Any] | Mask | None) -> "PointCloud | None":
        """The mask's point cloud, or None if it carries none.

        Accepts the plain dict predictions produced by worker jobs as well as
        the pydantic `Mask` model.
        """
        return PointCloud._build(mask, None)

    @staticmethod
    def from_object(obj: dict[str, Any] | Any) -> "PointCloud | None":
        """The point cloud of a predicted object's mask, or None if it has none.

        Records the object's bounding box, so `.at_source(x, y)` works.
        """
        if obj is None:
            return None
        if isinstance(obj, dict):
            mask = obj.get("mask")
            x, y, width, height = obj.get("x"), obj.get("y"), obj.get("width"), obj.get("height")
        else:
            mask = getattr(obj, "mask", None)
            x, y = getattr(obj, "x", None), getattr(obj, "y", None)
            width, height = getattr(obj, "width", None), getattr(obj, "height", None)
        if x is None or y is None or width is None or height is None:
            return PointCloud._build(mask, None)
        return PointCloud._build(mask, (float(x), float(y), float(width), float(height)))

    @staticmethod
    def from_depth(depth: dict[str, Any] | Depth | None,
                   source_width: float | None = None,
                   source_height: float | None = None) -> "PointCloud | None":
        """The scene point cloud of a frame level depth map, or None if it has none.

        Present when the Pop asked for `depthMap.toWorld`. Unlike a mask cloud
        this one spans the whole frame, so its grid is the depth map's rather
        than an object's bounding box - and `.at(i, j)` indexes map pixels, the
        same index that pixel's value sits at in `depth.values`.

        Pass the prediction's `source_width`/`source_height` to enable
        `.at_source(x, y)`; the frame is the box in this case.
        """
        if depth is None:
            return None
        if isinstance(depth, Depth):
            width, height, world = depth.width, depth.height, depth.world
        else:
            width, height, world = depth.get("width"), depth.get("height"), depth.get("world")
        if world is None:
            return None
        if width is None or height is None:
            raise ValueError("depth carries a 'world' point cloud but no width/height to shape it")
        box = None
        if source_width and source_height:
            box = (0.0, 0.0, float(source_width), float(source_height))
        return PointCloud(int(width), int(height), str(world), box)

    @staticmethod
    def from_prediction(prediction: dict[str, Any] | Any) -> list["PointCloud"]:
        """Every point cloud in a prediction, outermost object first.

        A list rather than DepthMap's single value because a cloud belongs to
        one object's mask, so a prediction carries as many as it has masked
        objects. Nested objects are included. A scene cloud, if the Pop asked
        for one, comes last - appended rather than inserted so an existing
        index into the object clouds keeps meaning what it did.
        """
        clouds: list[PointCloud] = []

        def walk(objects: Any) -> None:
            for obj in objects or []:
                cloud = PointCloud.from_object(obj)
                if cloud is not None:
                    clouds.append(cloud)
                walk(obj.get("objects") if isinstance(obj, dict) else getattr(obj, "objects", None))

        if prediction is None:
            return clouds
        walk(prediction.get("objects") if isinstance(prediction, dict)
             else getattr(prediction, "objects", None))
        if isinstance(prediction, dict):
            depth = prediction.get("depth")
            width, height = prediction.get("source_width"), prediction.get("source_height")
        else:
            depth = getattr(prediction, "depth", None)
            width = getattr(prediction, "source_width", None)
            height = getattr(prediction, "source_height", None)
        scene = PointCloud.from_depth(depth, width, height)
        if scene is not None:
            clouds.append(scene)
        return clouds

    @staticmethod
    def _build(mask: dict[str, Any] | Mask | None,
               box: tuple[float, float, float, float] | None) -> "PointCloud | None":
        if mask is None:
            return None
        if isinstance(mask, Mask):
            width, height, world = mask.width, mask.height, mask.world
        else:
            width, height, world = mask.get("width"), mask.get("height"), mask.get("world")
        if world is None:
            return None
        if width is None or height is None:
            raise ValueError("mask carries a 'world' point cloud but no width/height to shape it")
        return PointCloud(int(width), int(height), str(world), box)

    @property
    def array(self) -> np.ndarray:
        """The points as a float32 array of shape (height, width, 3); omitted points are NaN."""
        if self._array is None:
            try:
                raw = base64.b64decode(self._world, validate=True)
            except (TypeError, ValueError) as e:
                raise ValueError(f"invalid 'world' member: {e}") from e
            expected = self.width * self.height * _VALUES_PER_POINT * _BYTES_PER_VALUE
            if len(raw) != expected:
                raise ValueError(
                    f"'world' holds {len(raw)} bytes, expected {expected} for "
                    f"{self.width}x{self.height} xyz float32"
                )
            # NaN is the wire contract's omission sentinel, so unlike a depth map
            # there is nothing to reject here; +/-Infinity would be a broken
            # payload, but the worker omits rather than emitting one.
            self._array = np.frombuffer(raw, dtype="<f4").reshape(
                self.height, self.width, _VALUES_PER_POINT)
        return self._array

    @property
    def placed_mask(self) -> np.ndarray:
        """Boolean array of shape (height, width), True where the worker placed a point."""
        return np.asarray(~np.isnan(self.array).any(axis=2))

    @property
    def placed_points(self) -> np.ndarray:
        """Just the points the worker placed, as an (N, 3) float32 array.

        The shape a scatter plot or an export wants: the (height, width, 3) grid
        with the NaN holes dropped, and no mask to apply first.
        """
        points = self.array.reshape(-1, _VALUES_PER_POINT)
        return points[~np.isnan(points).any(axis=1)]

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        """Per axis (min, max) in metres over the placed points, or None if none were placed.

        The counterpart to DepthMap's finite_min/finite_max, which are one axis
        because a depth map has one value per pixel.
        """
        points = self.placed_points
        if points.size == 0:
            return None
        low = points.min(axis=0)
        high = points.max(axis=0)
        return ((float(low[0]), float(high[0])),
                (float(low[1]), float(high[1])),
                (float(low[2]), float(high[2])))

    def at(self, i: int, j: int) -> tuple[float, float, float] | None:
        """The world point for pixel (i, j), or None where the worker placed none.

        (i, j) indexes the cloud's own grid - the mask bitmap, or the depth map
        for a scene cloud: i is the column, j is the row.
        """
        if not (0 <= i < self.width and 0 <= j < self.height):
            raise IndexError(f"({i}, {j}) is outside a {self.width}x{self.height} cloud")
        point = self.array[j, i]
        if np.isnan(point).any():
            return None
        return float(point[0]), float(point[1]), float(point[2])

    def at_source(self, x: float, y: float) -> tuple[float, float, float] | None:
        """The world point for a source frame coordinate inside the cloud's box.

        Needs a cloud built with `from_object` or with `from_depth` and the
        frame size: the grid covers a box, and nothing else knows where it sits.
        """
        if self._box is None:
            raise ValueError("at_source needs a box; build with PointCloud.from_object or from_depth")
        box_x, box_y, box_width, box_height = self._box
        if box_width <= 0 or box_height <= 0:
            raise ValueError(f"invalid bounding box extent: width={box_width!r} height={box_height!r}")
        # the inverse of the worker's own grid-to-source transform, which samples
        # pixel centres: source = box + ((index + 0.5) / extent) * box_extent
        i = int((x - box_x) / box_width * self.width)
        j = int((y - box_y) / box_height * self.height)
        i = min(max(i, 0), self.width - 1)
        j = min(max(j, 0), self.height - 1)
        return self.at(i, j)

    def is_placed(self, i: int, j: int) -> bool:
        return self.at(i, j) is not None
