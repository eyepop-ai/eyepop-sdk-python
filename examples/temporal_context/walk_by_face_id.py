"""Keep the best face-ID vector of every person who walks by a camera.

People are detected and tracked on every frame. For each person, `select_crop`
picks the frame where their face is seen best and runs face detection and a
face-ID model on that one crop. The result is one vector per person, ready to
store or to compare: people whose vectors are similar are likely the same
person, seen on more than one walk-by.

    python walk_by_face_id.py --output faces.jsonl

See https://docs.eyepop.ai/developer-documentation/platform/temporal-context/example-face-id-from-walk-bys
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from eyepop import EyePopSdk
from eyepop.worker.worker_types import (
    CropForward,
    InferenceComponent,
    Pop,
    SelectForward,
    TrackingComponent,
)

VIDEO = Path(__file__).parent / "media" / "office_walk_by.mp4"

FACE_DETECTOR = "eyepop.person.face.short-range:latest"

POP = Pop(components=[
    # Efficient, every frame: find people...
    InferenceComponent(
        ability="eyepop.person.medium:latest",
        forward=CropForward(
            # ...and follow each one across frames, by appearance as well as motion.
            targets=[TrackingComponent(
                reidModel="eyepop.person.reid:latest",
                # Deep analysis, once per person: the frame where the face
                # detector is most confident wins, and a person it never finds
                # a face on is never selected.
                forward=SelectForward(
                    relevancyModel=FACE_DETECTOR,
                    minTrackLengthSeconds=2.0,  # a walk-by lasts longer; shorter tracks are fragments
                    targets=[InferenceComponent(
                        ability=FACE_DETECTOR,
                        # Cut the face out, rotate it upright, and embed it.
                        forward=CropForward(
                            boxPadding=1.25,
                            orientationTargetAngle=-90,
                            targets=[InferenceComponent(ability="eyepop.face-id.large:latest")],
                        ),
                    )],
                ),
            )],
        ),
    ),
])


def find_face_id(obj: dict[str, Any]) -> list[float] | None:
    """The face-ID vector, attached to the face object nested under the person."""
    for raw in obj.get("raw") or []:
        for tensor in raw.get("tensors") or []:
            if tensor.get("name") == "embedding":
                return tensor["data"][0]
    for child in obj.get("objects") or []:
        if found := find_face_id(child):
            return found
    return None


def group_by_face(faces: list[dict[str, Any]], threshold: float) -> list[list[tuple[dict[str, Any], float]]]:
    """Group walk-bys whose face-IDs are alike: each group is likely one person.

    Every walk-by in a group is more than `threshold` alike to every other one in it.
    Each walk-by is kept with its lowest cosine similarity to the rest of its group.
    """
    if not faces:
        return []
    vectors = np.array([face["faceId"] for face in faces])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors @ vectors.T  # cosine similarity of every pair of walk-bys

    groups: list[list[int]] = []
    for i in range(len(faces)):
        # The groups this walk-by is alike to all of, by its least-alike member.
        fits = [(similarity[i, group].min(), group) for group in groups]
        fits = [(sim, group) for sim, group in fits if sim > threshold]
        if fits:
            max(fits, key=lambda fit: fit[0])[1].append(i)
        else:
            groups.append([i])

    def lowest(i: int, group: list[int]) -> float:
        others = [j for j in group if j != i]
        return float(similarity[i, others].min()) if others else 1.0

    return [[(faces[i], lowest(i, group)) for i in group] for group in groups]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=str(VIDEO), help="a video file, or an rtsp:// or https:// URL")
    parser.add_argument("--output", help="write one JSON line per person: track id, time and face-ID vector")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="minimum cosine similarity for two walk-bys to count as the same person")
    args = parser.parse_args()

    faces: list[dict[str, Any]] = []

    with EyePopSdk.sync_worker(pop=POP) as endpoint:
        job = endpoint.load_from(args.video) if "://" in args.video else endpoint.upload(args.video)
        while result := job.predict():
            if not result.get("selected"):
                continue  # the live, per-frame picture; this example only needs selections
            person = result["objects"][0]
            face_id = find_face_id(person)
            if face_id is None:
                continue
            faces.append({"trackId": person["trackId"], "seconds": result["seconds"], "faceId": face_id})
            print(f"{result['seconds']:7.2f}s  person #{person['trackId']:<3} face-ID with {len(face_id)} dimensions")

    if args.output:
        with open(args.output, "w") as f:
            for face in faces:
                f.write(json.dumps(face) + "\n")

    people = group_by_face(faces, args.threshold)
    print(f"\n{len(faces)} walk-bys, {len(people)} different people")
    for person in people:
        if len(person) > 1:
            seen = ", ".join(f"#{face['trackId']} at {face['seconds']:.1f}s ({sim:.2f})" for face, sim in person)
            print(f"  walked by {len(person)} times: {seen}")


if __name__ == "__main__":
    main()
