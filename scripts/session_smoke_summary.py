from __future__ import annotations

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
