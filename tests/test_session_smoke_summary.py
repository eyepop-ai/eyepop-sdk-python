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
WORKFLOW_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "scripts"


def load_script(name: str) -> Any:
    return load_module(name, SCRIPTS_DIR / f"{name}.py")


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


session_smoke_summary = load_script("session_smoke_summary")
session_smoke = load_script("session_smoke")
workflow_summary = load_module(
    "ensure_session_smoke_summary",
    WORKFLOW_SCRIPTS_DIR / "ensure_session_smoke_summary.py",
)
ensure_summary = workflow_summary.ensure_summary
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
    def __init__(
        self,
        session_uuid: str = "session-12345678",
        upload_error: Exception | None = None,
        set_pop_session_uuid: str | None = None,
    ) -> None:
        self.compute_ctx = SimpleNamespace(session_uuid=session_uuid)
        self.upload_error = upload_error
        self.set_pop_session_uuid = set_pop_session_uuid

    async def set_pop(self, _: Any) -> None:
        # set_pop can move a transient endpoint to a different session.
        if self.set_pop_session_uuid:
            self.compute_ctx.session_uuid = self.set_pop_session_uuid

    async def upload(self, _: str) -> EmptyJob:
        if self.upload_error:
            raise self.upload_error
        return EmptyJob()


@asynccontextmanager
async def worker(endpoint: Endpoint) -> AsyncIterator[Endpoint]:
    yield endpoint


@pytest.fixture(autouse=True)
def no_preexisting_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    async def list_sessions(**_: str) -> set[str]:
        return set()

    monkeypatch.setattr(session_smoke, "list_session_uuids", list_sessions)


def record_deletes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    deleted: list[str] = []

    async def delete_session(**kwargs: str) -> dict[str, Any]:
        deleted.append(kwargs["session_uuid"])
        return {"ok": True, "result": "deleted"}

    monkeypatch.setattr(session_smoke, "delete_transient_session", delete_session)
    return deleted


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


@pytest.mark.asyncio
async def test_session_that_existed_before_the_run_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture user's sessions are shared; one the run adopted belongs to another client."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")

    async def list_sessions(**_: str) -> set[str]:
        return {"session-12345678"}

    monkeypatch.setattr(session_smoke, "list_session_uuids", list_sessions)
    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", lambda **_: worker(Endpoint()))
    deleted = record_deletes(monkeypatch)
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image), summary)

    assert deleted == []
    assert result["preexisting_session_count"] == 1
    assert result["cleanup"] == {"ok": True, "result": "reused", "session_uuid": "session-12345678"}


@pytest.mark.asyncio
async def test_deletes_every_session_the_run_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    endpoint = Endpoint(set_pop_session_uuid="session-87654321")
    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", lambda **_: worker(endpoint))
    deleted = record_deletes(monkeypatch)
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image), summary)

    assert deleted == ["session-12345678", "session-87654321"]
    assert result["session_uuids"] == ["session-12345678", "session-87654321"]
    assert result["cleanup"]["ok"] is True
    assert result["cleanup"]["result"] == "deleted"


@pytest.mark.asyncio
async def test_unlisted_sessions_are_never_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a pre-run listing the run cannot prove which session it created."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")

    async def list_sessions(**_: str) -> set[str]:
        raise RuntimeError("sessions list unavailable")

    monkeypatch.setattr(session_smoke, "list_session_uuids", list_sessions)
    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", lambda **_: worker(Endpoint()))
    deleted = record_deletes(monkeypatch)
    summary = new_summary(environment="production", requested_sdk_version="latest")

    result = await session_smoke.run_smoke(smoke_args(image), summary)

    assert deleted == []
    assert result["cleanup"]["ok"] is False
    assert result["cleanup"]["result"] == "ownership_unverified"
    assert result["cleanup"]["session_uuids"] == ["session-12345678"]


@pytest.mark.asyncio
async def test_opens_its_own_named_session_with_the_pop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Opening with the pop makes the SDK create a session instead of adopting the user's newest one."""
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    opened: list[dict[str, Any]] = []

    # Named parameters, as the real SDK declares them: the smoke detects support by signature.
    def async_worker(*, session_name: str | None = None, pop: Any = None, **kwargs: Any) -> Any:
        opened.append({**kwargs, "session_name": session_name, "pop": pop})
        return worker(Endpoint())

    monkeypatch.setattr(session_smoke.EyePopSdk, "async_worker", async_worker)
    record_deletes(monkeypatch)
    summary = new_summary(environment="production", requested_sdk_version="latest")

    await session_smoke.run_smoke(smoke_args(image), summary)

    assert opened[0]["session_name"] == "smoke-test"
    assert opened[0]["pop"] is not None
