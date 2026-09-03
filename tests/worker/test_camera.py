import math

import pytest
from pydantic import ValidationError

from eyepop.data.types.asset import RectangleArea
from eyepop.worker.camera import (
    Camera,
    CameraDistortion,
    CameraExtrinsics,
    CameraIntrinsics,
    Quaternion,
    Vector3d,
)
from eyepop.worker.worker_types import (
    ContourFinderComponent,
    ContourType,
    InferenceComponent,
    Pop,
    PopDepthMap,
    SourceDefaults,
)


def _intrinsics(**overrides) -> CameraIntrinsics:
    values = {"fx": 0.9, "fy": 1.6, "cx": 0.5, "cy": 0.5}
    values.update(overrides)
    return CameraIntrinsics(**values)


def test_camera_with_intrinsics():
    camera = Camera(intrinsics=_intrinsics())
    assert camera.intrinsics is not None
    assert camera.intrinsics.fx == 0.9


def test_camera_with_field_of_view_shorthand():
    assert Camera(hfovDegrees=72.0).hfovDegrees == 72.0


def test_both_lens_descriptions_are_rejected():
    # rejected rather than resolved by precedence: two descriptions of one lens
    # that disagree have no right answer
    with pytest.raises(ValidationError):
        Camera(intrinsics=_intrinsics(), hfovDegrees=72.0)


def test_neither_lens_description_is_rejected():
    with pytest.raises(ValidationError):
        Camera(distortion=CameraDistortion(k1=-0.28))


@pytest.mark.parametrize("hfov", [0.0, 180.0, -10.0, 360.0, math.nan, math.inf])
def test_field_of_view_outside_the_open_interval_is_rejected(hfov):
    with pytest.raises(ValidationError):
        Camera(hfovDegrees=hfov)


@pytest.mark.parametrize("hfov", [0.001, 60.0, 179.999])
def test_field_of_view_inside_the_open_interval_is_accepted(hfov):
    assert Camera(hfovDegrees=hfov).hfovDegrees == hfov


@pytest.mark.parametrize("focal", [0.0, -1.0, math.inf, math.nan])
def test_non_positive_or_non_finite_focal_length_is_rejected(focal):
    with pytest.raises(ValidationError):
        Camera(intrinsics=_intrinsics(fx=focal))
    with pytest.raises(ValidationError):
        Camera(intrinsics=_intrinsics(fy=focal))


@pytest.mark.parametrize("principal", [-0.01, 1.01, math.nan])
def test_principal_point_outside_the_frame_is_rejected(principal):
    with pytest.raises(ValidationError):
        Camera(intrinsics=_intrinsics(cx=principal))
    with pytest.raises(ValidationError):
        Camera(intrinsics=_intrinsics(cy=principal))


def test_principal_point_at_the_frame_edge_is_accepted():
    assert Camera(intrinsics=_intrinsics(cx=0.0, cy=1.0)) is not None


@pytest.mark.parametrize("coefficient", ["k1", "k2", "p1", "p2", "k3"])
def test_non_finite_distortion_is_rejected(coefficient):
    with pytest.raises(ValidationError):
        Camera(hfovDegrees=72.0, distortion=CameraDistortion(**{coefficient: math.inf}))


def test_distortion_defaults_to_a_rectilinear_lens():
    distortion = CameraDistortion()
    assert (distortion.k1, distortion.k2, distortion.p1, distortion.p2, distortion.k3) == (0.0,) * 5


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_non_finite_translation_is_rejected(axis):
    with pytest.raises(ValidationError):
        Camera(hfovDegrees=72.0,
               extrinsics=CameraExtrinsics(translation=Vector3d(**{axis: math.nan})))


def test_non_unit_quaternion_is_rejected():
    with pytest.raises(ValidationError):
        Camera(hfovDegrees=72.0, extrinsics=CameraExtrinsics(rotation=Quaternion(w=0.5, x=0.5)))


def test_non_finite_quaternion_is_rejected():
    # a non-finite component makes the norm non-finite, so the unit test catches
    # it without a separate finiteness check
    with pytest.raises(ValidationError):
        Camera(hfovDegrees=72.0, extrinsics=CameraExtrinsics(rotation=Quaternion(w=math.inf)))


def test_unit_quaternion_is_accepted():
    half_turn = 0.7071068
    camera = Camera(hfovDegrees=72.0,
                    extrinsics=CameraExtrinsics(rotation=Quaternion(w=half_turn, x=-half_turn)))
    assert camera.extrinsics is not None


def test_extrinsics_default_to_the_identity():
    rotation = Quaternion()
    assert (rotation.w, rotation.x, rotation.y, rotation.z) == (1.0, 0.0, 0.0, 0.0)
    translation = Vector3d()
    assert (translation.x, translation.y, translation.z) == (0.0, 0.0, 0.0)


def test_camera_serialises_to_the_wire_shape():
    camera = Camera(
        intrinsics=_intrinsics(),
        distortion=CameraDistortion(k1=-0.28, k2=0.07),
        extrinsics=CameraExtrinsics(rotation=Quaternion(w=1.0), translation=Vector3d(z=3.0)),
    )
    assert camera.model_dump(exclude_none=True) == {
        "intrinsics": {"fx": 0.9, "fy": 1.6, "cx": 0.5, "cy": 0.5},
        "distortion": {"k1": -0.28, "k2": 0.07, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "extrinsics": {"rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                       "translation": {"x": 0.0, "y": 0.0, "z": 3.0}},
    }


def test_field_of_view_camera_serialises_without_intrinsics():
    assert Camera(hfovDegrees=72.0).model_dump(exclude_none=True) == {"hfovDegrees": 72.0}


def test_camera_rejects_unknown_members():
    with pytest.raises(ValidationError):
        Camera(hfovDegrees=72.0, focalLength=1.0)  # type: ignore[call-arg]


def test_pop_defaults_carry_a_camera():
    pop = Pop(
        components=[InferenceComponent(ability="eyepop.person:latest")],
        defaults=SourceDefaults(camera=Camera(hfovDegrees=72.0), fps="10"),
    )
    dumped = pop.model_dump(exclude_none=True)
    assert dumped["defaults"] == {"camera": {"hfovDegrees": 72.0}, "fps": "10"}


def test_pop_defaults_carry_roi_and_motion_settings():
    defaults = SourceDefaults(
        roi=RectangleArea(x=1, y=2, width=3, height=4),
        motionDetect=True,
        motionGridX=10,
    )
    dumped = defaults.model_dump(exclude_none=True)
    assert dumped["roi"]["type"] == "rectangle"
    assert dumped["motionDetect"] is True
    assert dumped["motionGridX"] == 10


def test_pop_requests_world_coordinates():
    pop = Pop(
        components=[InferenceComponent(ability="eyepop.person:latest", toWorld=True)],
        depthMap=PopDepthMap(ability="eyepop.depth.anything-3:latest"),
    )
    dumped = pop.model_dump(exclude_none=True)
    assert dumped["depthMap"]["ability"] == "eyepop.depth.anything-3:latest"
    assert dumped["components"][0]["toWorld"] is True


def test_pop_by_ability_uuid():
    pop = Pop(components=[InferenceComponent(abilityUuid="a-uuid")],
              depthMap=PopDepthMap(abilityUuid="depth-uuid"))
    assert pop.model_dump(exclude_none=True)["depthMap"]["abilityUuid"] == "depth-uuid"


def test_pop_can_ask_for_the_whole_scene():
    # stands on its own: the map is a consumer in its own right, so no component
    # has to opt in for the pop to be complete
    pop = Pop(
        components=[InferenceComponent(ability="eyepop.person:latest")],
        depthMap=PopDepthMap(ability="eyepop.depth.anything-3:latest", toWorld=True),
    )
    dumped = pop.model_dump(exclude_none=True)
    assert dumped["depthMap"]["toWorld"] is True
    assert "toWorld" not in dumped["components"][0]


def test_a_pop_that_wants_no_enrichment_carries_none_of_it():
    pop = Pop(components=[InferenceComponent(ability="eyepop.person:latest")])
    dumped = pop.model_dump(exclude_none=True)
    assert "depthMap" not in dumped
    assert "defaults" not in dumped
    assert "toWorld" not in dumped["components"][0]


def test_to_world_is_accepted_on_every_component_type():
    # declared on the shared base, matching the instance's single PopComponent.
    # A component that runs no inference has no id for the worker to select on,
    # and is rejected when the Pop is compiled rather than here.
    component = ContourFinderComponent(contourType=ContourType.POLYGON, toWorld=True)
    assert component.toWorld is True
