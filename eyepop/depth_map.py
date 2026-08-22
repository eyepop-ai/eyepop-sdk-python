"""Decoding and sampling of frame-level depth maps.

Depth estimation abilities (e.g. eyepop.depth.*) attach a frame-level `depth`
member to each prediction: base64 of width*height little-endian float32 values
in row-major order, with the aspect ratio of the source frame and +Infinity
for sky pixels. Values are canonical metric depth (multiply by the camera's
focal length in pixels and divide by 300 for meters).
"""

import base64
import math
from typing import Any

import numpy as np

from eyepop.data.types.common import Depth


class DepthMap:
    """A decoded frame-level depth map.

    Create with `DepthMap.from_prediction(prediction)`; access the decoded
    values as a numpy float32 array of shape (height, width) via `.array`,
    or sample individual source frame coordinates with `.at(x, y, ...)`.
    """

    def __init__(self, width: int, height: int, values: str):
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid depth dimensions: width={width!r} height={height!r}")
        self.width = width
        self.height = height
        self._values = values
        self._array: np.ndarray | None = None

    @staticmethod
    def from_prediction(prediction: dict[str, Any] | Any) -> "DepthMap | None":
        """The prediction's depth map, or None if the prediction has none.

        Accepts the plain dict predictions produced by worker jobs as well as
        the pydantic `Prediction` model.
        """
        depth = prediction.get("depth") if isinstance(prediction, dict) else getattr(prediction, "depth", None)
        if depth is None:
            return None
        if isinstance(depth, Depth):
            return DepthMap(depth.width, depth.height, depth.values)
        return DepthMap(depth["width"], depth["height"], depth["values"])

    @property
    def array(self) -> np.ndarray:
        """The depth values as a float32 array of shape (height, width); sky pixels are +inf."""
        if self._array is None:
            try:
                raw = base64.b64decode(self._values, validate=True)
            except (TypeError, ValueError) as e:
                raise ValueError(f"invalid depth 'values' member: {e}") from e
            expected = self.width * self.height * 4
            if len(raw) != expected:
                raise ValueError(
                    f"depth 'values' holds {len(raw)} bytes, expected {expected} for {self.width}x{self.height} float32"
                )
            array = np.frombuffer(raw, dtype="<f4").reshape(self.height, self.width)
            # the wire contract reserves +inf for sky; NaN or -inf indicate a broken payload
            invalid = ~(np.isfinite(array) | np.isposinf(array))
            if invalid.any():
                raise ValueError(
                    f"depth 'values' contains {int(invalid.sum())} NaN or -Infinity values, "
                    "only finite values and +Infinity (sky) are allowed"
                )
            self._array = array
        return self._array

    @property
    def sky_mask(self) -> np.ndarray:
        """Boolean array of shape (height, width), True where the pixel is sky."""
        return np.isinf(self.array)

    @property
    def finite_min(self) -> float | None:
        """The smallest finite depth value, or None if the whole map is sky."""
        finite = self.array[np.isfinite(self.array)]
        return float(finite.min()) if finite.size else None

    @property
    def finite_max(self) -> float | None:
        """The largest finite depth value, or None if the whole map is sky."""
        finite = self.array[np.isfinite(self.array)]
        return float(finite.max()) if finite.size else None

    def at(self, x: float, y: float, source_width: float | None = None, source_height: float | None = None) -> float:
        """The depth value at map pixel (x, y), returning +inf for sky pixels.

        When source_width/source_height are given, (x, y) is a source frame
        coordinate and is mapped to the depth map proportionally.
        """
        if source_width and source_height:
            x = x * self.width / source_width
            y = y * self.height / source_height
        column = min(max(int(x), 0), self.width - 1)
        row = min(max(int(y), 0), self.height - 1)
        return float(self.array[row, column])

    def is_sky(self, x: float, y: float, source_width: float | None = None, source_height: float | None = None) -> bool:
        return math.isinf(self.at(x, y, source_width, source_height))
