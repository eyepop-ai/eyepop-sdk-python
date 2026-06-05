from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import aiohttp

from eyepop import EyePopSdk, __version__
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import DynamicComponent, InferenceComponent, Pop

DEFAULT_IMAGE = Path("tests/test.jpg")
DESCRIPTION = "Run a deterministic SDK transient-session smoke test."
ENV_URLS = {
    "production": "https://compute.eyepop.ai",
    "staging": "https://compute.staging.eyepop.xyz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--environment",
        choices=sorted(ENV_URLS.keys()),
        default=os.getenv("EYEPOP_ENV", "production"),
        help="EyePop environment to smoke.",
    )
    parser.add_argument(
        "--eyepop-url",
        default=os.getenv("EYEPOP_URL"),
        help="Compute API base URL. Defaults from --environment.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("EYEPOP_API_KEY"),
        help="EyePop API key. Defaults to EYEPOP_API_KEY.",
    )
    parser.add_argument(
        "--session-name",
        default=os.getenv("EYEPOP_SESSION_NAME"),
        help="Optional transient session name for run correlation.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Local image fixture to upload for inference.",
    )
    parser.add_argument(
        "--ability",
        default=os.getenv("EYEPOP_SMOKE_ABILITY", "eyepop.person:latest"),
        help="Pipeline ability alias for the smoke pop.",
    )
    parser.add_argument(
        "--expected-class",
        default=os.getenv("EYEPOP_SMOKE_EXPECTED_CLASS", "person"),
        help="Object class label required in the result.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=int(os.getenv("EYEPOP_SMOKE_MIN_OBJECTS", "1")),
        help="Minimum matching objects required.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.getenv("EYEPOP_SMOKE_MIN_CONFIDENCE", "0.5")),
        help="Minimum confidence for matching objects.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("EYEPOP_SMOKE_TIMEOUT_SECONDS", "600")),
        help="Overall smoke timeout.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path(os.getenv("EYEPOP_SMOKE_SUMMARY_JSON", "session-smoke-summary.json")),
        help="Path to write machine-readable run summary.",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip transient session deletion. Intended only for local debugging.",
    )
    return parser.parse_args()


def require_inputs(args: argparse.Namespace) -> None:
    if not args.api_key:
        raise ValueError("Missing EYEPOP_API_KEY")

    if args.min_objects < 1:
        raise ValueError("--min-objects must be at least 1")

    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0.0 and 1.0")

    if not args.image.is_file():
        raise FileNotFoundError(f"Image fixture does not exist: {args.image}")


def build_pop(args: argparse.Namespace) -> Pop:
    component = cast(
        DynamicComponent,
        InferenceComponent(
            ability=args.ability,
            categoryName=args.expected_class,
            modelUuid=None,
            model=None,
        ),
    )
    return Pop(
        components=[
            component
        ]
    )


def object_label(obj: dict[str, Any]) -> str:
    for key in ("classLabel", "category", "categoryName", "label", "name"):
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return ""


def matching_objects(result: dict[str, Any], expected_class: str, min_confidence: float) -> list[dict[str, Any]]:
    objects = result.get("objects") or []
    if not isinstance(objects, list):
        return []

    matches = []
    expected = expected_class.strip().lower()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        confidence = obj.get("confidence", 0)
        detected = object_label(obj).strip().lower()
        if detected == expected and isinstance(confidence, (int, float)) and confidence >= min_confidence:
            matches.append(obj)
    return matches


def summarize_predictions(
    predictions: list[dict[str, Any]],
    expected_class: str,
    min_confidence: float,
) -> dict[str, Any]:
    all_objects = []
    matches = []
    for result in predictions:
        objects = result.get("objects") or []
        if isinstance(objects, list):
            all_objects.extend(obj for obj in objects if isinstance(obj, dict))
        matches.extend(matching_objects(result, expected_class, min_confidence))

    return {
        "prediction_count": len(predictions),
        "object_count": len(all_objects),
        "matching_object_count": len(matches),
        "top_matches": [
            {
                "class": object_label(obj),
                "confidence": obj.get("confidence"),
                "x": obj.get("x"),
                "y": obj.get("y"),
                "width": obj.get("width"),
                "height": obj.get("height"),
            }
            for obj in matches[:5]
        ],
    }


def sdk_supports_session_name() -> bool:
    return "session_name" in inspect.signature(EyePopSdk.async_worker).parameters


def async_worker_kwargs(args: argparse.Namespace, eyepop_url: str) -> tuple[dict[str, Any], bool]:
    kwargs: dict[str, Any] = {
        "pop_id": "transient",
        "api_key": args.api_key,
        "eyepop_url": eyepop_url,
    }
    session_name_supported = sdk_supports_session_name()
    if args.session_name and session_name_supported:
        kwargs["session_name"] = args.session_name
    return kwargs, session_name_supported


async def delete_transient_session(api_key: str, eyepop_url: str, session_uuid: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    url = f"{eyepop_url.rstrip('/')}/v1/sessions/{session_uuid}"
    async with aiohttp.ClientSession() as session:
        async with session.delete(url, headers=headers) as response:
            body = await response.text()
            ok = response.status in (200, 202, 204, 404)
            return {
                "ok": ok,
                "status": response.status,
                "result": "deleted" if response.status != 404 else "already_absent",
                "body": body[:300] if body and not ok else "",
            }


async def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    require_inputs(args)

    eyepop_url = args.eyepop_url or ENV_URLS[args.environment]
    worker_kwargs, session_name_supported = async_worker_kwargs(args, eyepop_url)
    started = time.monotonic()
    session_uuid = ""
    summary: dict[str, Any] = {
        "ok": False,
        "environment": args.environment,
        "sdk_version": __version__,
        "eyepop_url": eyepop_url,
        "image": str(args.image),
        "ability": args.ability,
        "expected_class": args.expected_class,
        "min_objects": args.min_objects,
        "min_confidence": args.min_confidence,
        "session_name": args.session_name or "",
        "session_name_supported": session_name_supported,
        "session_name_applied": "session_name" in worker_kwargs,
        "session_uuid": "",
        "session_uuid_short": "",
        "cleanup": {"ok": True, "result": "not_started"},
    }

    try:
        async with EyePopSdk.async_worker(**worker_kwargs) as raw_endpoint:
            endpoint = cast(WorkerEndpoint, raw_endpoint)
            await endpoint.set_pop(build_pop(args))
            compute_ctx = getattr(endpoint, "compute_ctx", None)
            session_uuid = getattr(compute_ctx, "session_uuid", "") or ""
            summary["session_uuid"] = session_uuid
            summary["session_uuid_short"] = session_uuid[:8] if session_uuid else ""

            job = await endpoint.upload(str(args.image))
            predictions: list[dict[str, Any]] = []
            while result := await job.predict():
                if isinstance(result, dict):
                    predictions.append(result)

            prediction_summary = summarize_predictions(
                predictions,
                expected_class=args.expected_class,
                min_confidence=args.min_confidence,
            )
            summary.update(prediction_summary)

            smoke_ok = (
                prediction_summary["prediction_count"] > 0
                and prediction_summary["matching_object_count"] >= args.min_objects
            )
            if not smoke_ok:
                summary["error"] = (
                    f"Expected at least {args.min_objects} {args.expected_class!r} objects "
                    f"with confidence >= {args.min_confidence}; got "
                    f"{prediction_summary['matching_object_count']}"
                )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if session_uuid and not args.no_cleanup:
            try:
                summary["cleanup"] = await delete_transient_session(
                    api_key=args.api_key,
                    eyepop_url=eyepop_url,
                    session_uuid=session_uuid,
                )
            except Exception as exc:
                summary["cleanup"] = {
                    "ok": False,
                    "result": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        elif args.no_cleanup:
            summary["cleanup"] = {"ok": True, "result": "skipped"}
        else:
            summary["cleanup"] = {"ok": False, "result": "missing_session_uuid"}

    summary["duration_seconds"] = round(time.monotonic() - started, 3)
    summary["ok"] = "error" not in summary and bool(summary.get("cleanup", {}).get("ok", False))
    return summary


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


async def main() -> int:
    args = parse_args()
    summary: dict[str, Any]
    try:
        summary = await asyncio.wait_for(run_smoke(args), timeout=args.timeout_seconds)
    except Exception as exc:
        summary = {
            "ok": False,
            "environment": args.environment,
            "session_name": args.session_name or "",
            "error": f"{type(exc).__name__}: {exc}",
        }

    write_summary(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
