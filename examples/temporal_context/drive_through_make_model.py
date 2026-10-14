"""Name the make and model of every car entering and leaving a drive-through.

Vehicles are detected and tracked on every frame, and the tracks tell which way
each car drives: up the lane and away from the camera into the drive-through, or
toward the camera on the way out. For each car, `select_crop` picks the clearest
view of it and sends that one crop to a custom VLM ability, whose JSON response
schema returns the make, model and color as an object.

Create the ability in the dashboard first, then pass its alias:

    python drive_through_make_model.py --ability <namespace>.describe.vehicle-make-model:latest

See https://docs.eyepop.ai/developer-documentation/platform/temporal-context/example-car-make-and-model
"""
import argparse
from pathlib import Path
from typing import Any

from eyepop import EyePopSdk
from eyepop.worker.worker_types import (
    CropForward,
    InferenceComponent,
    Pop,
    SelectForward,
    TrackingComponent,
)

VIDEO = Path(__file__).parent / "media" / "drive_through.mp4"


def make_pop(ability: str) -> Pop:
    return Pop(components=[
        # Efficient, every frame: find vehicles...
        InferenceComponent(
            ability="eyepop.vehicle.medium:latest",
            objectAreaThreshold=0.01,  # ignore distant cars parked in the background
            forward=CropForward(
                # ...and follow each one across frames, even when its label
                # flickers between car and truck.
                targets=[TrackingComponent(
                    agnostic=True,
                    classHysteresis=True,
                    maxAgeSeconds=2.0,  # keep a car's track while the island hides it
                    # Deep analysis, once per selection: the VLM sees the
                    # clearest crop of each car, not every frame of it.
                    forward=SelectForward(
                        minTrackLengthSeconds=2.0,
                        intervalSeconds=10.0,
                        targets=[InferenceComponent(ability=ability)],
                    ),
                )],
            ),
        ),
    ])


def tracked(objects: list[dict[str, Any]] | None):
    """Every tracked object of a live prediction, however deep it is nested."""
    for obj in objects or []:
        if "trackId" in obj:
            yield obj
        yield from tracked(obj.get("objects"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ability", required=True,
                        help="alias of your make-and-model VLM ability, e.g. <namespace>.describe.vehicle-make-model:latest")
    parser.add_argument("--video", default=str(VIDEO), help="a video file, or an rtsp:// or https:// URL")
    args = parser.parse_args()

    frames = 0
    selections = 0
    cars: dict[int, dict[str, Any]] = {}
    # where each track was first and last seen: the vertical center of its box
    first_y: dict[int, float] = {}
    last_y: dict[int, float] = {}

    with EyePopSdk.sync_worker(pop=make_pop(args.ability)) as endpoint:
        if "://" in args.video:
            job = endpoint.load_from(args.video)
        else:
            job = endpoint.upload(args.video)

        while result := job.predict():
            if not result.get("selected"):
                # A live prediction: one per frame, the efficient side.
                frames += 1
                for obj in tracked(result.get("objects")):
                    center = obj["y"] + obj["height"] / 2
                    first_y.setdefault(obj["trackId"], center)
                    last_y[obj["trackId"]] = center
                continue

            # A selected prediction: the deep analysis of one car, about a past frame.
            selections += 1
            car = result["objects"][0]
            track_id = car["trackId"]
            # the VLM's structured answer, shaped by its response schema
            answer = dict((car.get("details") or [{}])[0])
            # up the frame is away from the camera, into the drive-through
            answer["direction"] = "entering" if last_y[track_id] < first_y[track_id] else "exiting"
            cars[track_id] = answer
            print(f"{result['seconds']:7.2f}s  car #{track_id:<3} {answer['direction']:<9} "
                  f"{answer.get('make', '?')} {answer.get('model', '?')} ({answer.get('color', '?')})")

    entering = sum(1 for car in cars.values() if car["direction"] == "entering")
    print(f"\n{frames} frames analyzed live; {selections} VLM calls named {len(cars)} cars, "
          f"{entering} entering and {len(cars) - entering} exiting.")


if __name__ == "__main__":
    main()
