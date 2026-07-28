from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_summary_helper() -> Any:
    helper_path = REPOSITORY_ROOT / "scripts" / "session_smoke_summary.py"
    spec = importlib.util.spec_from_file_location("session_smoke_summary", helper_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    helper = load_summary_helper()
    existing = helper.read_summary(path)
    summary = existing or helper.new_summary(
        environment=environment,
        requested_sdk_version=requested_sdk_version,
        sdk_version=resolved_sdk_version,
        session_name=session_name,
    )
    defaults = helper.new_summary(
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
        summary["github_run_url"] = helper.github_run_url()
    if not summary.get("github_run_id"):
        summary["github_run_id"] = os.getenv("GITHUB_RUN_ID", "")

    failed_step = step_failure(steps)
    if not existing:
        phase, message = failed_step or (
            "harness",
            "Smoke workflow did not produce a valid JSON summary.",
        )
        helper.record_failure(summary, phase, message)
    elif failed_step and not summary.get("error"):
        phase, message = failed_step
        helper.record_failure(summary, phase, message)

    elapsed = round(max(0.0, time.time() - started_at), 3)
    if not isinstance(summary.get("duration_seconds"), (int, float)) or summary["duration_seconds"] <= 0:
        summary["duration_seconds"] = elapsed

    cleanup = summary.get("cleanup")
    if not isinstance(cleanup, dict):
        summary["cleanup"] = {"ok": False, "result": "invalid_summary"}
    helper.write_summary(path, summary)
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
