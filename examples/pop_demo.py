import argparse
import ast
import asyncio
import base64
import json
import logging
import os
import sys
from argparse import Namespace
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from PIL import Image
from pybars import Compiler
from pydantic import TypeAdapter
from relay_example import relay_http_source, relay_rtsp_source
from webui import webui

from eyepop import EyePopSdk, Job
from eyepop.data.data_types import TranscodeMode
from eyepop.data.types.asset import Area, RectangleArea
from eyepop.visualize import EyePopWorldPlot, labelled_world_points
from eyepop.worker.camera import Camera, CameraIntrinsics
from eyepop.worker.worker_types import (
    BaseComponent,
    ComponentParams,
    ContourFinderComponent,
    ContourType,
    CropForward,
    ForwardComponent,
    FullForward,
    InferenceComponent,
    MotionDetectConfig,
    MotionModel,
    Pop,
    PopDepthMap,
    TrackingComponent,
)

# The depth ability used when --to-world is asked for on its own. Must
# be a metric one: a 'relative' map is accepted and silently yields no
# coordinates, because relative depth is scale- AND shift-invariant, so a cloud
# recovered from it would be distorted rather than merely unscaled.
DEFAULT_DEPTH_ABILITY = 'eyepop.depth.large:latest'

load_dotenv()

logging.basicConfig(level=logging.INFO)

log = logging.getLogger('eyepop.example')

script_dir = os.path.dirname(__file__)


pop_examples = {
    "noop": Pop(components=[]),
    "vehicles": Pop(components=[
        InferenceComponent(
            ability='eyepop.vehicle:latest',
            categoryName="vehicles",
            confidenceThreshold=0.8,
            forward=CropForward(
                targets=[TrackingComponent(
                    maxAgeSeconds=5.0,
                    motionModel=MotionModel.CONSTANT_VELOCITY,
                    agnostic=True,
                    classHysteresis=True,
                ), InferenceComponent(
                    ability='eyepop.vehicle.license-plate:latest',
                    topK=1,
                    forward=CropForward(
                        targets=[InferenceComponent(
                            ability='eyepop.text.recognize.landscape:latest',
                            categoryName="license-plate"
                        )]
                    )
                )]
            )
        )
    ]),

    "person": Pop(components=[
        InferenceComponent(
            ability='eyepop.person:latest',
            categoryName="person"
        )
    ]),

    "2d-body-points": Pop(components=[
        InferenceComponent(
            ability='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    ability='eyepop.person.2d-body-points:latest',
                    categoryName="2d-body-points",
                    confidenceThreshold=0.25
                )]
        ))
    ]),

    "faces": Pop(components=[
        InferenceComponent(
            ability='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    ability='eyepop.person.face.short-range:latest',
                    categoryName="2d-face-points",
                    forward=CropForward(
                        boxPadding=1.5,
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            ability='eyepop.person.face-mesh:latest',
                            categoryName="3d-face-mesh"
                        )]
                    )
                )]
            )
        )
    ]),

    "hands": Pop(components=[
        InferenceComponent(
            ability='eyepop.person:latest',
            categoryName="person",
            forward=CropForward(
                maxItems=128,
                boxPadding=0.25,
                targets=[InferenceComponent(
                    ability='eyepop.person.palm:latest',
                    forward=CropForward(
                        includeClasses=["hand circumference"],
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            ability='eyepop.person.3d-hand-points:latest',
                            categoryName="3d-hand-points"
                        )]
                    )
                )]
            )
        )
    ]),

    "3d-body-points": Pop(components=[
        InferenceComponent(
            ability='eyepop.person:latest',
            categoryName="person",
            objectAreaThreshold=0.01,
            forward=CropForward(
                boxPadding=0.5,
                targets=[InferenceComponent(
                    ability='eyepop.person.pose:latest',
                    hidden=True,
                    forward=CropForward(
                        boxPadding=0.5,
                        orientationTargetAngle=-90.0,
                        targets=[InferenceComponent(
                            ability='eyepop.person.3d-body-points.heavy:latest',
                            categoryName="3d-body-points",
                            confidenceThreshold=0.1
                        )]
                    )
                )]
            )
        )
    ]),

    "text": Pop(components=[
        InferenceComponent(
            ability='eyepop.text:latest',
            categoryName="text",
            confidenceThreshold=0.7,
            forward=CropForward(
                maxItems=128,
                targets=[InferenceComponent(
                    ability='eyepop.text.recognize.landscape:latest',
                    confidenceThreshold=0.1
                )]
            )
        )
    ]),

    "sam1": Pop(components=[
        InferenceComponent(
            ability='eyepop.sam.small:latest',
            id=1,
            forward=FullForward(
                targets=[ContourFinderComponent(
                    contourType=ContourType.POLYGON,
                    areaThreshold=0.005
                )]
            )
        )
    ]),

    "sam2": Pop(components=[
        InferenceComponent(
            ability="eyepop.sam2.encoder.tiny:latest",
            id=1,
            hidden=True,
            forward=FullForward(
                targets=[InferenceComponent(
                    ability='eyepop.sam2.decoder:latest',
                    forward=FullForward(
                        targets=[ContourFinderComponent(
                            contourType=ContourType.POLYGON,
                            areaThreshold=0.005
                        )]
                    )
                )]
            )
        )
    ]),
    "image-contents": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.image-contents:latest',
        )
    ]),
    "localize-objects": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.localize-objects:latest',
        )
    ]),
    "depth": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.depth.large:latest',
        )
    ]),
    "localize-objects-plus": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.localize-objects:latest',
            params={
                "prompts": [{"prompt": "person"}]
            },
            forward=CropForward(
                targets=[InferenceComponent(
                    ability='eyepop.image-contents:latest',
                    params={
                        "prompts": [{"prompt": "hair color blond"},{"prompt": "hair color brown"}]
                    }
                )],
            )
        )
    ]),
    "localize-objects-t4": Pop(components=[
        InferenceComponent(
            id=1,
            ability='eyepop.localize-objects:latest',
            params={
                "prompts": [{"prompt": "person"}]
            },
            forward=CropForward(
                targets=[InferenceComponent(
                    ability='eyepop.image-contents-t4:latest',
                    params={
                        "prompts": [{"prompt": "shirt color?"}]
                    }
                )],
            )
        )
    ]),
}

def list_of_points(arg: str) -> list[dict[str, Any]]:
    points = []
    points_as_tuples = ast.literal_eval(f'[{arg}]')
    for tuple in points_as_tuples:
        points.append({
            "x": tuple[0],
            "y": tuple[1]
        })
    return points


def list_of_boxes(arg: str) -> list[dict[str, Any]]:
    boxes = []
    boxes_as_tuples = ast.literal_eval(f'[{arg}]')
    for tuple in boxes_as_tuples:
        boxes.append({
            "topLeft": {
                "x": tuple[0],
                "y": tuple[1]
            },
            "bottomRight": {
                "x": tuple[2],
                "y": tuple[3]
            }
        })
    return boxes


def rectangle_roi(arg: str) -> Area:
    roi = ast.literal_eval(arg)

    return RectangleArea(
        x=roi[0],
        y=roi[1],
        width=roi[2],
        height=roi[3],
    )


def camera_intrinsics(arg: str) -> CameraIntrinsics:
    fx, fy, cx, cy = ast.literal_eval(arg)

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)


def camera_from_args(camera_args: Namespace) -> Camera | None:
    """The source calibration, or None to let the worker assume a field of view.

    Assuming one is a development scaffold: for canonical metric depth the guess
    cancels out of X and Y and survives only in Z, so lateral measurements stay
    exact while every distance along the optical axis is wrong by however wrong
    the guess was.
    """
    if camera_args.camera_intrinsics is not None:
        return Camera(intrinsics=camera_args.camera_intrinsics)
    if camera_args.camera_hfov_degrees is not None:
        return Camera(hfovDegrees=camera_args.camera_hfov_degrees)
    return None


def request_world_coordinates(components: list[BaseComponent]) -> int:
    """Opt every component that can be enriched into world coordinates.

    Returns how many were opted in, walking nested forward targets so this works
    with the deeply composed examples as well as the flat ones.

    Only a component that runs its own inference can honour it, which is what
    gives it an id for the worker to select on; the converter rejects it on the
    others. A hidden component is skipped rather than rejected - its predictions
    never reach the response, so back-projecting them would be paid for and
    thrown away - but its forward targets are still walked, which is what an
    encoder-then-detector composition needs.
    """
    count = 0
    for component in components:
        if isinstance(component, (InferenceComponent, TrackingComponent)):
            if not (isinstance(component, InferenceComponent) and component.hidden):
                component.toWorld = True
                count += 1
        if component.forward is not None and component.forward.targets:
            count += request_world_coordinates(component.forward.targets)
    return count


def add_world_coordinates_to_pop(original: Pop, world_args: Namespace) -> Pop:
    """Return a copy of the pop that asks for world coordinates.

    Copied rather than mutated because the examples are shared module level
    objects, and a demo that edits one in place would be a trap for the next
    reader.
    """
    if not (world_args.to_world or world_args.depth_map_to_world):
        return original

    pop = original.model_copy(deep=True)
    depth_map = PopDepthMap(toWorld=True if world_args.depth_map_to_world else None)
    if world_args.depth_map_ability_uuid:
        depth_map.abilityUuid = world_args.depth_map_ability_uuid
    else:
        depth_map.ability = world_args.depth_map_ability or DEFAULT_DEPTH_ABILITY
    pop.depthMap = depth_map

    enriched = request_world_coordinates(pop.components) if world_args.to_world else 0
    if enriched == 0 and not world_args.depth_map_to_world:
        # the converter rejects this rather than silently doing nothing, so say
        # so here where the reason is obvious
        log.warning("no component in this pop can carry world coordinates, so the depth "
                    "ability has nothing to enrich")
    else:
        log.info("requesting world coordinates for %d component(s)%s via %s", enriched,
                 " and the whole scene" if world_args.depth_map_to_world else "",
                 depth_map.abilityUuid or depth_map.ability)
    return pop


def summarize_world_coordinates(prediction: dict[str, Any]) -> str | None:
    """One line on how many points came back placed, or None if none did.

    A point the worker could not place carries no world members at all - sky,
    outside the depth map, no usable map - so counting them is what tells a
    calibration that worked from one that quietly did not.
    """
    placed = 0
    unplaced = 0

    def count_points(points: list[dict[str, Any]]) -> None:
        nonlocal placed, unplaced
        for point in points:
            if point.get("worldZ") is not None:
                placed += 1
            else:
                unplaced += 1

    def walk(obj: dict[str, Any]) -> None:
        for keypoints in obj.get("keyPoints") or []:
            count_points(keypoints.get("points") or [])
        count_points(obj.get("outline") or [])
        for contour in obj.get("contours") or []:
            count_points(contour.get("points") or [])
            for cutout in contour.get("cutouts") or []:
                count_points(cutout)
        for nested in obj.get("objects") or []:
            walk(nested)

    # a prediction carries the same point bearing members an object does, so one
    # walk from the root covers both; walking its objects again would count the
    # whole tree twice
    walk(prediction)

    if placed == 0 and unplaced == 0:
        return None
    return f"world coordinates: {placed} point(s) placed, {unplaced} not"


def collect_world_points(prediction: dict[str, Any], into: list[Any]) -> None:
    """Accumulate everything in a prediction that carries world coordinates.

    Key points, outlines, contours and mask clouds alike. Only the placed points
    are kept, and only as arrays: a mask covers its object's whole bounding box,
    so a cloud is mostly holes for anything that is not rectangular, and holding
    the raw results across a video would cost far more than the scatter can draw.
    """
    into.extend(labelled_world_points(prediction))



def replace_binary_members(value):
    """Summarize large binary members (depth values, mask bitmaps).

    Replaces base64 payloads with a short description instead of dumping
    megabytes of base64 to the console.
    """
    if isinstance(value, dict):
        replaced = {k: replace_binary_members(v) for k, v in value.items()}
        if isinstance(replaced.get("values"), str) and "width" in replaced and "height" in replaced:
            replaced["values"] = f"<{replaced['width']}x{replaced['height']} base64 float32, {len(value['values'])} chars>"
        if isinstance(replaced.get("bitmap"), str) and "width" in replaced and "height" in replaced:
            replaced["bitmap"] = f"<{replaced['width']}x{replaced['height']} base64 bitmap, {len(value['bitmap'])} chars>"
        if isinstance(replaced.get("world"), str) and "width" in replaced and "height" in replaced:
            replaced["world"] = f"<{replaced['width']}x{replaced['height']} base64 xyz float32, {len(value['world'])} chars>"
        return replaced
    if isinstance(value, list):
        return [replace_binary_members(v) for v in value]
    return value


def add_optional_tracking_to_component(component: ForwardComponent, tracking_args: Namespace):
    if tracking_args.tracking:
        tracking_component = TrackingComponent()
        if tracking_args.tracking_reid_model is not None:
            tracking_component.reidModel = tracking_args.tracking_reid_model
        if tracking_args.tracking_max_age is not None:
            tracking_component.maxAgeSeconds = tracking_args.tracking_max_age
        if tracking_args.tracking_iou_threshold is not None:
            tracking_component.iouThreshold = tracking_args.tracking_iou_threshold
        if tracking_args.tracking_sim_threshold is not None:
            tracking_component.simThreshold = tracking_args.tracking_sim_threshold
        if tracking_args.tracking_agnostic is not None:
            tracking_component.agnostic = tracking_args.tracking_agnostic
        if tracking_args.tracking_motion_model is not None:
            tracking_component.motionModel = MotionModel(tracking_args.tracking_motion_model)
        component.forward = CropForward(
            targets=[tracking_component]
        )


parser = argparse.ArgumentParser(
                    prog='Pop examples',
                    description='Demonstrates the composition of a Pop',
                    epilog='.')
parser.add_argument('-d', '--dump', required=False, help="dump composable pop definition to stdout", default=False, action="store_true")
parser.add_argument('-l', '--local-path', required=False, type=str, default=False, help="run the inference on a local file, or all files on a directory")
parser.add_argument('--sample-count', required=False, type=int, help="When running the inference on a directory, sample trhis number of images")
parser.add_argument('-a', '--asset-uuid', required=False, type=str, default=False, help="run the inference on an asset by its Uuid")
parser.add_argument('-u', '--url', required=False, type=str, default=False, help="run the inference on a remote Url")
parser.add_argument('-proxy', '--proxy-url', required=False, type=str, default=False, help="Resolve the given URL and proxy the content stream")
parser.add_argument('-p', '--pop', required=False, type=str, help="run this pop", choices=list(pop_examples.keys()))
parser.add_argument('-s', '--session', required=False, type=str, help="Use an existing a session uuid and do not set a pop, use the sessions as preconfigured", default=None)
parser.add_argument('-m', '--model-uuid', required=False, type=str, action="append", help="run this model(s) by uuid")
parser.add_argument('-ma', '--model-alias', required=False, type=str, action="append", help="run this model(s) by its tagged alias")
parser.add_argument('-ms1', '--model-uuid-sam1', required=False, type=str, help="run this model by uuid and compose with SAM1 (EfficientSAM) and Contour Finder")
parser.add_argument('-ms2', '--model-uuid-sam2', required=False, type=str, help="run this model by uuid and compose with SAM2 and Contour Finder")
parser.add_argument('-po', '--points', required=False, type=list_of_points, help="List of POIs as coordinates like (x1, y1), (x2, y2) in the original image coordinate system")
parser.add_argument('-bo', '--boxes', required=False, type=list_of_boxes, help="List of POIs as boxes like (left1, top1, right1, bottom1), (left1, top1, right1, bottom1) in the original image coordinate system")
parser.add_argument('-sp', '--single-prompt', required=False, type=str, help="Single prompt to pass as parameter")
parser.add_argument('-sl', '--single-label', required=False, type=str, help="Single label to use as result for single prompt to pass as parameter")
parser.add_argument('-pr', '--prompt', required=False, type=str, help="Prompt to pass as parameter", action="append")
parser.add_argument('-v', '--visualize', required=False, help="show rendered output", default=False, action="store_true")
parser.add_argument('-o', '--output', required=False, help="print results to stdout", default=False, action="store_true")
parser.add_argument('-ds', '--dataset-uuid', required=False, type=str, help="Ingest all assets into a dataset uuid", default=None)
parser.add_argument('-tk', '--top-k', required=False, type=int, help="For --model-uuid and -model-alias apply this top-k filter", default=None)
parser.add_argument('-ct', '--confidence-threshold', required=False, type=float, help="For --model-uuid and -model-alias apply this confidence threshold filter", default=None)
parser.add_argument('--fps', required=False, type=str, help="For --model-uuid and -model-alias apply this target fps - e.g. 15/1", default=None)
parser.add_argument('-vl', '--video-chunk-length-seconds', required=False, type=float, help="For --model-uuid and -model-alias apply this video chunk length", default=None)
parser.add_argument('-vo', '--video-chunk-overlap', required=False, type=float, help="For --model-uuid and -model-alias apply this video chunk overlap", default=None)

# Optional tracking for simple pops my model uuid oder model alias
parser.add_argument('--tracking', required=False, help="Track objects in videos", default=False, action="store_true")
parser.add_argument('--tracking-reid-model', required=False, help="Use re-id model uuid for tracking", default=None, type=str)
parser.add_argument('--tracking-agnostic', required=False, help="Track objects class-agnostic", default=False, action="store_true")
parser.add_argument('--tracking-max-age', required=False, help="Max age in seconds for unmatched tracks", default=None, type=float)
parser.add_argument('--tracking-iou-threshold', required=False, help="IoU threshold to match tracks", default=None, type=float)
parser.add_argument('--tracking-sim-threshold', required=False, help="Similarity threshold to match tracks by re-id", default=None, type=float)
parser.add_argument('--tracking-motion-model', required=False, help="Pick a motion model for tracking, one of 'random_walk', 'constant_velocity' or 'constant_acceleration'", default=None, type=str)

# Optional motion detection parameters
parser.add_argument('--motion-detect', required=False, help="Skip video frames w/o detected motion", default=False, action="store_true")

# Optional global ROI parameters
parser.add_argument('--roi', required=False, type=rectangle_roi, help="Rectangular ROI as (x, y, width, height)")

parser.add_argument('-w', '--to-world', required=False, default=False, action="store_true",
                    help="Translate this pop's point based predictions into world coordinates in metres, "
                         "back-projected through a depth map. Works with any of the example pops")
parser.add_argument('--depth-map-to-world', required=False, default=False, action="store_true",
                    help="Back-project the depth map itself, so the results carry a point cloud of the "
                         "whole scene rather than one per segmented object. Also what reveals the map: "
                         "without it the depth branch stays out of the response. Stands on its own, "
                         "with or without --to-world")
parser.add_argument('--depth-map-ability', required=False, type=str, default=None,
                    help=f"Depth ability supplying the map to back-project through, default "
                         f"'{DEFAULT_DEPTH_ABILITY}'. Must be a metric one; a 'relative' ability is "
                         f"accepted and silently yields no coordinates")
parser.add_argument('--depth-map-ability-uuid', required=False, type=str, default=None,
                    help="Depth ability by uuid, instead of --depth-map-ability")
parser.add_argument('--camera-hfov-degrees', required=False, type=float, default=None,
                    help="Source's horizontal field of view in (0, 180). Without a calibration the "
                         "worker assumes 60 degrees, which stretches world coordinates along the "
                         "optical axis by however wrong that guess is")
parser.add_argument('--camera-intrinsics', required=False, type=camera_intrinsics, default=None,
                    help="Source's normalized intrinsics as (fx, fy, cx, cy), fractions of the frame "
                         "rather than pixels. Mutually exclusive with --camera-hfov-degrees")
parser.add_argument('-vw', '--visualize-world', required=False, default=False, action="store_true",
                    help="Scatter everything in the results that carries world coordinates - key "
                         "points, outlines, contours, mask point clouds and the scene cloud - into "
                         "a 3D plot, in metres. Needs --to-world or --depth-map-to-world to fill them")
parser.add_argument('--world-max-points', required=False, type=int,
                    default=EyePopWorldPlot.DEFAULT_MAX_POINTS,
                    help="Point budget for --visualize-world, shared across every series; sparse "
                         "ones such as key points are drawn whole regardless")

# Optional caching media for post-processing on the worker
parser.add_argument('-mc', '--media-cache-seconds', required=False, type=int, help="Cache most recent X seconds of media for post-processing on the worker", default=None)


main_args = parser.parse_args()

if not main_args.local_path and not main_args.url and not main_args.asset_uuid and not main_args.proxy_url:
    print("Need something to run inference on; pass either --url or --local-path or --asset-uuid or --proxy-url")
    parser.print_help()
    sys.exit(1)

if (not main_args.pop
        and not main_args.model_uuid
        and not main_args.model_alias
        and not main_args.model_uuid_sam1
        and not main_args.model_uuid_sam2
        and not main_args.session):
    print("Need something do do, pass either --pop or --model-uuid or --model-alias or --model-uuid-sam1 "
          "or --model-uuid-sam2 (or start with a preconfigured session with --session)")
    parser.print_help()
    sys.exit(1)

if main_args.pop:
    pop = pop_examples[main_args.pop]
elif main_args.model_uuid:
    pop = Pop(components=[
        InferenceComponent(
            id=i+1,
            abilityUuid=uuid,
        ) for i, uuid in enumerate(main_args.model_uuid)
    ])
    if main_args.top_k is not None:
        for c in pop.components:
            c.topK = main_args.top_k

    if main_args.confidence_threshold is not None:
        for c in pop.components:
            c.confidenceThreshold = main_args.confidence_threshold

    if main_args.video_chunk_length_seconds is not None:
        for c in pop.components:
            c.videoChunkLengthSeconds = main_args.video_chunk_length_seconds

    if main_args.video_chunk_overlap is not None:
        for c in pop.components:
            c.videoChunkOverlap = main_args.video_chunk_overlap

    add_optional_tracking_to_component(pop.components[0], main_args)
elif main_args.model_alias:
    pop = Pop(components=[
        InferenceComponent(
            id=i+1,
            ability=alias,
        ) for i, alias in enumerate(main_args.model_alias)
    ])

    if main_args.top_k is not None:
        for c in pop.components:
            c.topK = main_args.top_k

    if main_args.confidence_threshold is not None:
        for c in pop.components:
            c.confidenceThreshold = main_args.confidence_threshold

    if main_args.video_chunk_length_seconds is not None:
        for c in pop.components:
            c.videoChunkLengthSeconds = main_args.video_chunk_length_seconds

    if main_args.video_chunk_overlap is not None:
        for c in pop.components:
            c.videoChunkOverlap = main_args.video_chunk_overlap

    add_optional_tracking_to_component(pop.components[0], main_args)
elif main_args.model_uuid_sam1:
    pop = Pop(components=[
        InferenceComponent(
            abilityUuid=main_args.model_uuid_sam1,
            forward=CropForward(
                targets=[InferenceComponent(
                    ability='eyepop.sam.small:latest',
                    # forward=FullForward(
                    #     targets=[ContourFinderComponent(
                    #         contourType=ContourType.POLYGON,
                    #         areaThreshold=0.005
                    #     )]
                    # )
                )]
            )
        )
    ])
elif main_args.model_uuid_sam2:
    pop = Pop(components=[
        InferenceComponent(
            ability="eyepop.sam2.encoder.tiny:latest",
            hidden=True,
            forward=FullForward(
                targets=[InferenceComponent(
                    abilityUuid=main_args.model_uuid_sam2,
                    forward=CropForward(
                        targets=[InferenceComponent(
                            ability='eyepop.sam2.decoder:latest',
                            forward=FullForward(
                                targets=[ContourFinderComponent(
                                    contourType=ContourType.POLYGON,
                                    areaThreshold=0.005
                                )]
                            )
                        )]
                    )
                )]
            )
        )
    ])
elif main_args.session:
    pop = None
elif main_args.depth_map_to_world:
    # a complete pop on its own: the depth map is the only consumer, so there is
    # nothing for a component to do and none has to be named
    pop = Pop(components=[])
else:
    raise ValueError("pop or model required (or a preconfigured session, "
                     "or --depth-map-to-world for the scene alone)")

if main_args.camera_intrinsics is not None and main_args.camera_hfov_degrees is not None:
    # rejected rather than resolved by precedence: two descriptions of one lens
    # that disagree have no right answer
    print("Pass either --camera-intrinsics or --camera-hfov-degrees, not both")
    sys.exit(1)

if ((main_args.depth_map_ability or main_args.depth_map_ability_uuid)
        and not (main_args.to_world or main_args.depth_map_to_world)):
    print("--depth-map-ability needs --to-world or --depth-map-to-world; a depth ability nothing asked "
          "to use is rejected as a bad pop rather than silently doing nothing")
    sys.exit(1)

if main_args.to_world or main_args.depth_map_to_world:
    if pop is None:
        print("world coordinates cannot be added to a preconfigured session; pass a pop or a model")
        sys.exit(1)
    pop = add_world_coordinates_to_pop(pop, main_args)

camera = camera_from_args(main_args)
if (main_args.to_world or main_args.depth_map_to_world) and camera is None:
    log.warning("no camera calibration supplied, so the worker assumes a 60 degree horizontal field "
                "of view; lateral measurements are exact but depth is only as right as that guess. "
                "Pass --camera-hfov-degrees to turn the guess into a measurement")

params = None
if main_args.points:
    params = [
        ComponentParams(componentId=1, values={
          "roi": {
              "points": main_args.points
          }
        })
    ]
elif main_args.boxes:
    params = [
        ComponentParams(componentId=1, values={
            "roi": {
                "boxes": main_args.boxes
            }
        })
    ]
elif main_args.prompt is not None and len(main_args.prompt) > 0:
    params = [
        ComponentParams(componentId=1, values={
            "prompts": [{"prompt": p} for p in main_args.prompt]
        })
    ]
elif main_args.single_prompt is not None:
    params = [
        ComponentParams(componentId=1, values={
            "prompt": main_args.single_prompt,
            "label": main_args.single_label if main_args.single_label else main_args.single_prompt
        })
    ]

async def main(args) -> tuple[dict[str, Any] | None, str | None, list[Any]]:
    visualize_prediction = None
    visualize_path = None
    example_image_src = None
    world_points: list[Any] = []
    motion_detect = MotionDetectConfig(motionGap=1) if args.motion_detect else None

    if args.dump and pop:
        print("Pop:", pop.model_dump_json(exclude_none=True, indent=2))
        if params:
            print("Params:", TypeAdapter(list[ComponentParams]).dump_json(params, exclude_none=True, indent=2).decode("utf-8"))

    async with EyePopSdk.async_worker(dataset_uuid=args.dataset_uuid, session_uuid=args.session) as endpoint:
        if pop:
            await endpoint.set_pop(pop)

        if args.local_path:
            if not os.path.exists(args.local_path):
                log.warning(f"local path {args.local_path} does not exist")
                sys.exit(1)
            if os.path.isfile(args.local_path):
                local_files = [args.local_path]
            else:
                local_files = []

                for f in os.listdir(args.local_path):
                    if args.sample_count is not None and args.sample_count <= len(local_files):
                        break
                    local_file = os.path.join(args.local_path, f)
                    if os.path.isfile(local_file):
                        local_files.append(local_file)
            jobs = []
            async def on_ready(job: Job, path: str):
                nonlocal visualize_prediction
                nonlocal visualize_path
                while result := await job.predict():
                    visualize_prediction = result
                    if args.visualize_world:
                        collect_world_points(result, world_points)
                    visualize_path = path
                    if args.output:
                        print(path, json.dumps(replace_binary_members(result), indent=2))
                        if (world := summarize_world_coordinates(result)) is not None:
                            print(path, world)
            for local_file in local_files:
                job = await endpoint.upload(
                    local_file,
                    params=params,
                    motion_detect=motion_detect,
                    roi=args.roi,
                    fps=args.fps,
                    camera=camera,
                    media_cache_seconds=args.media_cache_seconds
                )
                jobs.append(on_ready(job, local_file))
            await asyncio.gather(*jobs)
            if args.visualize and visualize_prediction is not None:
                image = Image.open(visualize_path)
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                example_image_src = f"data:image/png;base64, {base64.b64encode(buffer.getvalue()).decode()}"
        elif args.url:
            job = await endpoint.load_from(
                args.url,
                params=params,
                motion_detect=motion_detect,
                roi=args.roi,
                fps=args.fps,
                camera=camera,
                media_cache_seconds=args.media_cache_seconds
            )
            while result := await job.predict():
                visualize_prediction = result
                if args.visualize_world:
                    collect_world_points(result, world_points)
                if args.output:
                    print(json.dumps(replace_binary_members(result), indent=2))
                    if (world := summarize_world_coordinates(result)) is not None:
                        print(world)
            if args.visualize:
                example_image_src = args.url
        elif args.proxy_url:
            if args.proxy_url.startswith("http:") or args.proxy_url.startswith("https:"):
                async for result in relay_http_source(
                        source_url=args.proxy_url,
                        endpoint=endpoint,
                        params=params,
                        motion_detect=motion_detect,
                        roi=args.roi,
                        fps=args.fps,
                        camera=camera,
                ):
                    visualize_prediction = result
                    if args.visualize_world:
                        collect_world_points(result, world_points)
                    if args.output:
                        print(json.dumps(replace_binary_members(result), indent=2))
                        if (world := summarize_world_coordinates(result)) is not None:
                            print(world)
                if args.visualize:
                    example_image_src = args.url
            elif args.proxy_url.startswith("rtsp:"):
                async for result in relay_rtsp_source(
                        source_url=args.proxy_url,
                        endpoint=endpoint,
                        params=params,
                        motion_detect=motion_detect,
                        roi=args.roi,
                        fps=args.fps,
                        camera=camera,
                ):
                    visualize_prediction = result
                    if args.visualize_world:
                        collect_world_points(result, world_points)
                    if args.output:
                        print(json.dumps(replace_binary_members(result), indent=2))
                        if (world := summarize_world_coordinates(result)) is not None:
                            print(world)
                if args.visualize:
                    example_image_src = args.url
            else:
                print(f"unsupported protocol in proxy URL {args.proxy_url}")
                sys.exit(1)

        elif args.asset_uuid:
            job = await endpoint.load_asset(
                args.asset_uuid,
                params=params,
                motion_detect=motion_detect,
                roi=args.roi,
                fps=args.fps,
                camera=camera,
                media_cache_seconds=args.media_cache_seconds
            )
            while result := await job.predict():
                visualize_prediction = result
                if args.visualize_world:
                    collect_world_points(result, world_points)
                if args.output:
                    print(json.dumps(replace_binary_members(result), indent=2))
                    if (world := summarize_world_coordinates(result)) is not None:
                        print(world)
            if args.visualize:
                async with EyePopSdk.dataEndpoint(is_async=True) as dataEndpoint:
                    buffer = await dataEndpoint.download_asset(
                        args.asset_uuid,
                        transcode_mode=TranscodeMode.image_original_size
                    ).read()
                    example_image_src = f"data:image/jpeg;base64, {base64.b64encode(buffer).decode()}"
    return visualize_prediction, example_image_src, world_points

visualize_result, example_image_src, result_world_points = asyncio.run(main(main_args))

if main_args.visualize_world:
    if not result_world_points:
        print("nothing in the results carries world coordinates: --visualize-world needs "
              "--translate-to-world, a metric depth ability, and a pop whose predictions have "
              "points to place")
    else:
        import matplotlib.pyplot as plt

        world_axes = plt.figure(figsize=(9, 8)).add_subplot(projection='3d')
        world_plot = EyePopWorldPlot(world_axes)
        drawn = world_plot.series(result_world_points, max_points=main_args.world_max_points)
        total = sum(len(entry.points) for entry in result_world_points)
        world_plot.finish(title=f"{len(result_world_points)} series, {drawn} of {total} points")
        plt.show()
if main_args.visualize:
    with open(os.path.join(script_dir, 'viewer.html')) as file:
        compiler = Compiler()
        html_template = compiler.compile(file.read())

    preview = html_template({
        'image_src': example_image_src,
        'result_json': json.dumps(visualize_result)
    })
    window = webui.window()
    window.set_root_folder('.')
    window.show(preview)
    webui.wait()
