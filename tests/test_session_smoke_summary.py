from __future__ import annotations

import importlib.util
import sys
import time
from argparse import Namespace
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


session_smoke_summary = load_script("session_smoke_summary")
session_smoke = load_script("session_smoke")
ensure_summary = session_smoke_summary.ensure_summary
finalize_summary = session_smoke_summary.finalize_summary
new_summary = session_smoke_summary.new_summary
read_summary = session_smoke_summary.read_summary


def smoke_args(image: Path, **overrides: Any) -> Namespace:
    values = {
        "environment": "production",
        "eyepop_url": "https://compute.eyepop.ai",
        "api_key": "test-key",
        "session_name": "smoke-test",
        "image": image,
        "ability": "eyepop.person:latest",
        "expected_class": "person",
        "min_objects": 1,
        "min_confidence": 0.5,
        "timeout_seconds": 60,
        "summary_json": image.parent / "summary.json",
        "no_cleanup": False,
    }
    values.update(overrides)
    return Namespace(**values)


@asynccontextmanager
async def failing_worker() -> AsyncIterator[Any]:
    raise RuntimeError("token exchange unavailable")
    yield


class EmptyJob:
    async def predict(self) -> None:
        return None


class Endpoint:
    def __init__(self, session_uuid: str = "session-12345678", upload_error: Exception | None = None) -> None:
        self.compute_ctx = SimpleNamespace(session_uuid=session_uuid)
        self.upload_error = upload_error

    async def set_pop(self, _: Any) -> None:
        return None

    async def upload(self, _: str) -> EmptyJob:
        if self.upload_error:
            raise self.upload_error
        return EmptyJob()


@asynccontextmanager
async def worker(endpoint: Endpoint) -> AsyncIterator[Endpoint]:
    yield endpoint


@pytest.mark.parametrize(
    ("failed_step", "expected_phase"),
    [("install_sdk", "sdk_install"), ("resolve_sdk", "sdk_resolution")],
)
def test_sdk_install_or_resolution_failure_writes_fallback_summary(
    tmp_path: Path, failed_step: str, expected_phase: str
) -> None:
    path = tmp_path / "summary.json"

    summary = ensure_summary(
        path=path,
        environment="production",
        requested_sdk_version="3.17.2.dev221301",
        session_name="smoke-production-1-1",
        resolved_sdk_version="",
        steps={failed_step: "failure", "smoke": "skipped"},
        started_at=time.time() - 2,
    )

    assert read_summary(path) == summary
    assert summary["phase"] == expected_phase
    assert summary["failure_kind"] == "setup"
    assert summary["requested_sdk_version"] == "3.17.2.dev221301"
    assert summary["prediction_count"] is None
    assert summary["matching_object_count"] is None
    assert summary["cleanup"] == {"ok": False, "result": "not_started"}
    assert summary["duration_seconds"] > 0


@pytest.mark.asyncio
async def test_auth_setup_failure_has_complete_summary(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image, api_key=""), summary)
    finalize_summary(result, time.monotonic() - 1)

    assert result["phase"] == "validation"
    assert result["failure_kind"] == "setup"
    assert result["error"]
    assert result["cleanup"] == {"ok": True, "result": "not_required"}


@pytest.mark.asyncio
async def test_session_creation_failure_has_complete_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", lambda **_: failing_worker())
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image), summary)
    finalize_summary(result, time.monotonic() - 1)

    assert result["phase"] == "session_creation"
    assert result["failure_kind"] == "infrastructure"
    assert result["error"]
    assert result["session_uuid"] == ""
    assert result["cleanup"] == {"ok": True, "result": "not_required"}


@pytest.mark.asyncio
async def test_prediction_failure_preserves_created_session_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    deleted: list[str] = []
    endpoint = Endpoint(upload_error=RuntimeError("prediction worker disconnected"))
    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", lambda **_: worker(endpoint))

    async def delete_session(**kwargs: str) -> dict[str, Any]:
        deleted.append(kwargs["session_uuid"])
        return {"ok": True, "result": "deleted"}

    monkeypatch.setattr(session_smoke, "delete_transient_session", delete_session)
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image), summary)
    finalize_summary(result, time.monotonic() - 1)

    assert result["phase"] == "prediction"
    assert result["prediction_count"] is None
    assert result["matching_object_count"] is None
    assert deleted == ["session-12345678"]
    assert result["cleanup"] == {"ok": True, "result": "deleted"}


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_primary_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", lambda **_: worker(Endpoint()))

    async def delete_session(**_: str) -> dict[str, Any]:
        raise RuntimeError("cleanup backend unavailable")

    monkeypatch.setattr(session_smoke, "delete_transient_session", delete_session)
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image), summary)
    finalize_summary(result, time.monotonic() - 1)

    assert result["phase"] == "assertion"
    assert result["failure_kind"] == "assertion"
    assert result["error"]
    assert result["cleanup"]["result"] == "error"
    assert result["cleanup"]["error"]
