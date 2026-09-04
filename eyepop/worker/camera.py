"""Camera calibration for a source, as accepted by the worker's `setSource`.

A calibration is what turns a depth map into measurable world coordinates.
Without one the worker falls back to an assumed horizontal field of view, which
is a development scaffold rather than something to ship: for canonical metric
depth the guess cancels out of X and Y and survives only in Z, so lateral
measurements stay exact while every distance along the optical axis is wrong by
however wrong the guess was.

Mirrors what the instance API accepts, and is validated here so a bad
calibration is a local error rather than a 400 halfway through an upload. The
worker validates authoritatively; these two are meant to agree.
"""

import math

from pydantic import BaseModel, ConfigDict, model_validator

# How far a rotation's norm may sit from 1.0 and still be taken for a rotation.
# Loose enough to survive a float round trip through JSON, tight enough to catch
# an un-normalised or garbage one. Matches the instance and the worker.
QUATERNION_TOLERANCE = 1e-3


def _is_finite(value: float) -> bool:
    return not (math.isnan(value) or math.isinf(value))


class CameraIntrinsics(BaseModel):
    """The pinhole parameters, normalised to the frame rather than given in pixels.

    Normalised so one calibration survives a resolution change: a camera
    calibrated at 1920x1080 and later streamed at 1280x720 keeps working. Pixel
    values carrying no calibration resolution are wrong by exactly the
    resolution ratio, and nothing downstream can detect it.

    `fx`/`fy` are the focal length as a fraction of the frame's width and
    height; `cx`/`cy` are the principal point in the same normalisation.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    model_config = ConfigDict(extra='forbid')


class CameraDistortion(BaseModel):
    """The OpenCV Brown-Conrady coefficients. All zero is a rectilinear lens."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    model_config = ConfigDict(extra='forbid')


class Quaternion(BaseModel):
    """A rotation, w first. Must be a unit quaternion."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    model_config = ConfigDict(extra='forbid')


class Vector3d(BaseModel):
    """A translation in metres."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    model_config = ConfigDict(extra='forbid')


class CameraExtrinsics(BaseModel):
    """The camera's pose, expressed camera -> world.

    As `P_world = R(rotation) * P_camera + translation`, so the rotation
    carries the camera's axes onto the world's and the translation is where the
    camera itself sits. The world frame is Z up with the ground plane at Z = 0.

    This is the inverse of what cv2.solvePnP returns; a caller holding its
    rvec/tvec must invert both halves (R = R_cv.T, t = -R_cv.T @ t_cv), and
    t_cv is *not* the camera position.

    Left unset, the pose is the identity and world coordinates come back in the
    camera frame instead - OpenCV convention, X right, Y down, Z forward.
    """

    rotation: Quaternion | None = None
    translation: Vector3d | None = None
    model_config = ConfigDict(extra='forbid')


class Camera(BaseModel):
    """One source's calibration.

    Exactly one of `intrinsics` and `hfovDegrees` describes the lens. Both is
    rejected rather than resolved by precedence - two descriptions of one lens
    that disagree have no right answer - and neither is rejected too, since
    defaulting a focal length would be inventing a lens.

    `hfovDegrees` is the shorthand for an operator who knows their lens's field
    of view but not its calibration matrix. It assumes square pixels and a
    centred principal point, so it approximates a real calibration rather than
    replacing one - but it describes the actual lens rather than the worker's
    assumed fallback. It composes with distortion and extrinsics exactly as
    intrinsics do.
    """

    intrinsics: CameraIntrinsics | None = None
    hfovDegrees: float | None = None
    distortion: CameraDistortion | None = None
    extrinsics: CameraExtrinsics | None = None
    model_config = ConfigDict(extra='forbid')

    @model_validator(mode='after')
    def _validate(self) -> "Camera":
        if self.intrinsics is not None and self.hfovDegrees is not None:
            raise ValueError("camera.intrinsics and camera.hfovDegrees are alternatives, supply one")
        if self.intrinsics is None and self.hfovDegrees is None:
            raise ValueError("camera requires either camera.intrinsics or camera.hfovDegrees")

        if self.hfovDegrees is not None:
            # the upper bound excludes +inf and every comparison with nan is
            # false, so this one test covers the non-finite cases too
            if not 0 < self.hfovDegrees < 180:
                raise ValueError("camera.hfovDegrees must be a horizontal field of view in (0, 180) degrees")
        else:
            assert self.intrinsics is not None
            # note +inf passes a bare `> 0`, so finiteness is its own test
            for name, focal in (("fx", self.intrinsics.fx), ("fy", self.intrinsics.fy)):
                if not focal > 0 or not _is_finite(focal):
                    raise ValueError(f"camera.intrinsics.{name} must be a positive, finite focal length")
            for name, principal, extent in (("cx", self.intrinsics.cx, "width"),
                                            ("cy", self.intrinsics.cy, "height")):
                if not 0 <= principal <= 1:
                    raise ValueError(f"camera.intrinsics.{name} must be within [0, 1] of the frame {extent}")

        if self.distortion is not None:
            for name in ("k1", "k2", "p1", "p2", "k3"):
                if not _is_finite(getattr(self.distortion, name)):
                    raise ValueError(f"camera.distortion.{name} must be a finite number")

        if self.extrinsics is not None:
            if self.extrinsics.translation is not None:
                for axis in ("x", "y", "z"):
                    if not _is_finite(getattr(self.extrinsics.translation, axis)):
                        raise ValueError(f"camera.extrinsics.translation.{axis} must be a finite number")
            # a non-finite component makes the norm non-finite, so the unit test
            # below rejects it without a separate finiteness check
            rotation = self.extrinsics.rotation
            if rotation is not None:
                norm = math.sqrt(rotation.w ** 2 + rotation.x ** 2 + rotation.y ** 2 + rotation.z ** 2)
                if not abs(norm - 1.0) <= QUATERNION_TOLERANCE:
                    raise ValueError("camera.extrinsics.rotation must be a unit quaternion")

        return self
