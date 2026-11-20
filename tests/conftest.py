import pytest
from aioresponses import aioresponses as aioresponses_mocker


@pytest.fixture
def aioresponses():
    """Fixture to mock aiohttp requests."""
    with aioresponses_mocker() as m:
        yield m


@pytest.fixture(autouse=True)
def worker_session_mode(monkeypatch: pytest.MonkeyPatch):
    # Pipeline mode is the default; the tests outside test_pipeline_mode.py cover the worker session.
    monkeypatch.setenv("EYEPOP_PIPELINE_MODE", "false")
