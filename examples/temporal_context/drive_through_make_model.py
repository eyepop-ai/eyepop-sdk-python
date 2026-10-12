"""Name the make and model of every car that passes a drive-through lane.

Vehicles are detected and tracked on every frame. For each car, `select_crop`
picks the clearest view of it and sends that one crop to a custom VLM ability,
whose JSON response schema returns the make, model and color as an object.

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
            ability="eyepop.vehicle.large:latest",
            confidenceThreshold=0.5,
            objectAreaThreshold=0.01,  # ignore distant cars parked in the background
            forward=CropForward(
                includeClasses=["car", "truck"],
                # ...and follow each one across frames, even when its label
                # flickers between car and truck.
                targets=[TrackingComponent(
                    agnostic=True,
                    # Deep analysis, once per selection: the VLM sees the
                    # clearest crop of each car, not every frame of it.
                    forward=SelectForward(
                        minTrackLengthSeconds=1.0,
                        intervalSeconds=10.0,
                        boxPadding=1.1,
                        targets=[InferenceComponent(ability=ability)],
                    ),
                )],
            ),
        ),
    ])


def find_details(obj: dict[str, Any]) -> dict[str, Any] | None:
    """The VLM's structured response, nested somewhere under the selected car."""
    for details in obj.get("details") or []:
        if isinstance(details, dict):
            return details
    for child in obj.get("objects") or []:
        if found := find_details(child):
            return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ability", required=True,
                        help="alias of your make-and-model VLM ability, e.g. <namespace>.describe.vehicle-make-model:latest")
    parser.add_argument("--video", default=str(VIDEO), help="a video file, or an rtsp:// or https:// URL")
    args = parser.parse_args()

    frames = 0
    selections = 0
    cars: dict[int, dict[str, Any]] = {}

    with EyePopSdk.sync_worker(pop=make_pop(args.ability)) as endpoint:
        if "://" in args.video:
            job = endpoint.load_from(args.video)
        else:
            job = endpoint.upload(args.video)

        while result := job.predict():
            if not result.get("selected"):
                # A live prediction: one per frame, the efficient side.
                frames += 1
                continue

            # A selected prediction: the deep analysis of one car, about a past frame.
            selections += 1
            car = result["objects"][0]
            answer = find_details(car) or {}
            cars[car["trackId"]] = answer
            print(f"{result['seconds']:7.2f}s  car #{car['trackId']:<3} "
                  f"{answer.get('make', '?')} {answer.get('model', '?')} ({answer.get('color', '?')})")

    print(f"\n{frames} frames analyzed live; {selections} VLM calls named {len(cars)} cars.")


if __name__ == "__main__":
    main()
