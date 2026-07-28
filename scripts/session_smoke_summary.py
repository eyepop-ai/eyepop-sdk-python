from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


def github_run_url() -> str:
    server_url = os.getenv("GITHUB_SERVER_URL", "")
    repository = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if server_url and repository and run_id:
        return f"{server_url}/{repository}/actions/runs/{run_id}"
    return ""


def failure_kind(phase: str) -> str:
    if phase == "assertion":
        return "assertion"
    if phase in {"harness", "smoke_harness"}:
        return "harness"
    if phase == "cleanup":
        return "cleanup"
    if phase in {"validation", "checkout", "setup", "sdk_install", "sdk_resolution"}:
        return "setup"
    return "infrastructure"


def new_summary(
    *,
    environment: str,
    requested_sdk_version: str,
    sdk_version: str = "",
    session_name: str = "",
    phase: str = "harness",
) -> dict[str, Any]:
    return {
        "ok": False,
        "environment": environment,
        "requested_sdk_version": requested_sdk_version,
        "sdk_version": sdk_version,
        "phase": phase,
        "failure_kind": failure_kind(phase),
        "error": "",
        "github_run_id": os.getenv("GITHUB_RUN_ID", ""),
        "github_run_url": github_run_url(),
        "session_name": session_name,
        "session_uuid": "",
        "session_uuid_short": "",
        "prediction_count": None,
        "matching_object_count": None,
        "cleanup": {"ok": False, "result": "not_started"},
        "duration_seconds": 0.0,
    }


def record_failure(summary: dict[str, Any], phase: str, error: str) -> None:
    summary["phase"] = phase
    summary["failure_kind"] = failure_kind(phase)
    summary["error"] = error
    summary["ok"] = False


def finalize_summary(summary: dict[str, Any], started_at: float) -> dict[str, Any]:
    summary["duration_seconds"] = round(time.monotonic() - started_at, 3)
    session_uuid = summary.get("session_uuid")
    if session_uuid and not summary.get("session_uuid_short"):
        summary["session_uuid_short"] = str(session_uuid)[:8]

    cleanup = summary.get("cleanup")
    if not isinstance(cleanup, dict):
        cleanup = {"ok": False, "result": "invalid_summary"}
        summary["cleanup"] = cleanup

    if not summary.get("error") and not cleanup.get("ok", False):
        cleanup_error = cleanup.get("error") or cleanup.get("result", "unknown error")
        record_failure(summary, "cleanup", f"Cleanup failed: {cleanup_error}")

    summary["ok"] = not summary.get("error") and bool(cleanup.get("ok", False))
    if summary["ok"]:
        summary["phase"] = "complete"
        summary["failure_kind"] = ""
    return summary


def write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def read_summary(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def step_failure(steps: Mapping[str, str]) -> tuple[str, str] | None:
    phases = (
        ("checkout", "checkout", "Repository checkout failed before the smoke harness started."),
        ("setup_uv", "setup", "Smoke runner setup failed before the SDK was installed."),
        ("install_sdk", "sdk_install", "SDK installation failed before the smoke harness started."),
        ("resolve_sdk", "sdk_resolution", "Installed SDK version resolution failed before the smoke harness started."),
        ("smoke", "smoke_harness", "Smoke harness exited before writing a valid JSON summary."),
    )
    for step, phase, message in phases:
        if steps.get(step) in {"failure", "cancelled", "timed_out"}:
            return phase, message
    return None


def ensure_summary(
    *,
    path: Path,
    environment: str,
    requested_sdk_version: str,
    session_name: str,
    resolved_sdk_version: str,
    steps: Mapping[str, str],
    started_at: float,
) -> dict[str, Any]:
    existing = read_summary(path)
    summary = existing or new_summary(
        environment=environment,
        requested_sdk_version=requested_sdk_version,
        sdk_version=resolved_sdk_version,
        session_name=session_name,
    )
    defaults = new_summary(
        environment=environment,
        requested_sdk_version=requested_sdk_version,
        sdk_version=resolved_sdk_version,
        session_name=session_name,
    )
    for key, value in defaults.items():
        summary.setdefault(key, value)

    if not summary.get("sdk_version") and resolved_sdk_version:
        summary["sdk_version"] = resolved_sdk_version
    if not summary.get("requested_sdk_version"):
        summary["requested_sdk_version"] = requested_sdk_version
    if not summary.get("github_run_url"):
        summary["github_run_url"] = github_run_url()
    if not summary.get("github_run_id"):
        summary["github_run_id"] = os.getenv("GITHUB_RUN_ID", "")

    failed_step = step_failure(steps)
    if not existing:
        phase, message = failed_step or (
            "harness",
            "Smoke workflow did not produce a valid JSON summary.",
        )
        record_failure(summary, phase, message)
    elif failed_step and not summary.get("error"):
        phase, message = failed_step
        record_failure(summary, phase, message)

    elapsed = round(max(0.0, time.time() - started_at), 3)
    if not isinstance(summary.get("duration_seconds"), (int, float)) or summary["duration_seconds"] <= 0:
        summary["duration_seconds"] = elapsed

    cleanup = summary.get("cleanup")
    if not isinstance(cleanup, dict):
        summary["cleanup"] = {"ok": False, "result": "invalid_summary"}
    write_summary(path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensure a sessions smoke JSON summary exists.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--requested-sdk-version", required=True)
    parser.add_argument("--session-name", default="")
    parser.add_argument("--resolved-sdk-version-file", type=Path)
    parser.add_argument("--started-at-file", type=Path)
    parser.add_argument("--step", action="append", default=[])
    return parser.parse_args()


def read_text(path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def main() -> int:
    args = parse_args()
    steps = dict(item.split("=", 1) for item in args.step if "=" in item)
    try:
        started_at = float(read_text(args.started_at_file))
    except ValueError:
        started_at = time.time()
    summary = ensure_summary(
        path=args.summary_json,
        environment=args.environment,
        requested_sdk_version=args.requested_sdk_version,
        session_name=args.session_name,
        resolved_sdk_version=read_text(args.resolved_sdk_version_file),
        steps=steps,
        started_at=started_at,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
