"""Pipeline mode: a refused socket (HTTP 503 or 429) is waited through, honouring Retry-After, and then raises."""
import logging
from collections.abc import AsyncIterator

import aiohttp
import pytest

from eyepop import EyePopSdk
from eyepop.exceptions import (
    PipelineCommandException,
    PipelineOverloadedException,
    PipelineSocketClosedException,
)
from eyepop.settings import Settings, settings
from eyepop.worker import pipeline_backend
from eyepop.worker.pipeline_backend import _refusal_wait
from eyepop.worker.worker_endpoint import WorkerEndpoint
from tests.worker.pipeline_fakes import FakeComputeAPI, FakePipelineAPI, prediction


@pytest.fixture(autouse=True)
def no_environment(monkeypatch: pytest.MonkeyPatch):
    for name in ("EYEPOP_URL", "EYEPOP_API_KEY", "EYEPOP_SECRET_KEY", "EYEPOP_ACCESS_TOKEN", "EYEPOP_POP_ID",
                 "EYEPOP_SESSION_UUID", "EYEPOP_PIPELINE_MODE", "EYEPOP_LOCAL_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
async def pipeline() -> AsyncIterator[FakePipelineAPI]:
    async with FakePipelineAPI() as fake:
        yield fake


@pytest.fixture
async def compute(pipeline: FakePipelineAPI) -> AsyncIterator[FakeComputeAPI]:
    async with FakeComputeAPI(pipeline.url) as fake:
        yield fake


class Clock:
    """The upgrade's clock and sleep, replaced: a sleep moves the clock and takes no time."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(pipeline_backend, "_sleep", fake.sleep)
    monkeypatch.setattr(pipeline_backend, "_monotonic", fake.monotonic)
    return fake


def worker(compute: FakeComputeAPI, **options) -> WorkerEndpoint:
    options.setdefault("api_key", "the api key")
    return EyePopSdk.async_worker(eyepop_url=compute.url, is_pipeline_mode=True, **options)


def test_the_wait_defaults_to_ten_minutes_and_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    assert Settings().pipeline_wait_timeout == 600.0
    monkeypatch.setenv("EYEPOP_PIPELINE_WAIT_TIMEOUT", "5")
    assert Settings().pipeline_wait_timeout == 5.0


async def test_the_upgrade_waits_the_retry_after_of_each_refusal(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(503, {"Retry-After": "3"}), (503, {"Retry-After": "7"})]
    async with worker(compute):
        pass
    assert clock.sleeps == [3.0, 7.0]
    assert len(pipeline.sockets) == 1


async def test_a_429_is_waited_through_like_a_503(compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(429, {"Retry-After": "4"})]
    async with worker(compute):
        pass
    assert clock.sleeps == [4.0]
    assert len(pipeline.sockets) == 1


async def test_without_retry_after_the_wait_is_a_jittered_backoff(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(503, {})] * 6
    async with worker(compute):
        pass
    assert len(clock.sleeps) == 6
    for wait, longest in zip(clock.sleeps, (1, 2, 4, 8, 10, 10), strict=True):
        assert longest / 2 <= wait <= longest


def test_the_backoff_jitter_stays_within_bounds():
    for refusals, longest in enumerate((1, 2, 4, 8, 10, 10, 10)):
        waits = [_refusal_wait({}, refusals) for _ in range(200)]
        assert all(longest / 2 <= wait <= longest for wait in waits)
        assert len(set(waits)) > 1, "no jitter"


def test_retry_after_wins_over_the_backoff_and_an_unreadable_one_falls_back_to_it():
    assert _refusal_wait({"Retry-After": "30"}, 5) == 30.0
    assert _refusal_wait({"Retry-After": "0"}, 0) == 1.0, "a floor, so 0 is no request storm"
    assert _refusal_wait({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, 0) <= 1.0
    assert _refusal_wait({"Retry-After": "-3"}, 0) <= 1.0


async def test_a_socket_refused_for_the_whole_wait_raises_overloaded(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 20.0)
    pipeline.upgrade_answers = [(503, {"Retry-After": "8"})] * 10
    with pytest.raises(PipelineOverloadedException) as refused:
        async with worker(compute):
            pass
    # The last wait is cut to what is left, so the last try is at the limit.
    assert clock.sleeps == [8.0, 8.0, 4.0]
    assert "stayed at capacity for 20 seconds" in str(refused.value)
    assert refused.value.waited_secs == 20.0
    assert refused.value.error == "overloaded"
    assert isinstance(refused.value, PipelineCommandException)
    assert isinstance(refused.value.__cause__, aiohttp.ClientResponseError)
    assert len(pipeline.upgrade_answers) == 10 - 4, "four tries: the first and one after each wait"
    assert pipeline.sockets == []


async def test_a_wait_of_zero_does_not_wait(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 0.0)
    pipeline.upgrade_answers = [(503, {"Retry-After": "1"})]
    with pytest.raises(PipelineOverloadedException):
        async with worker(compute):
            pass
    assert clock.sleeps == []
    assert pipeline.upgrade_answers == []


@pytest.mark.parametrize("status", [400, 403, 404, 500, 502])
async def test_other_statuses_raise_at_once(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, status: int):
    pipeline.upgrade_answers = [(status, {"Retry-After": "5"})]
    with pytest.raises(aiohttp.ClientResponseError) as refused:
        async with worker(compute):
            pass
    assert refused.value.status == status
    assert clock.sleeps == []


async def test_a_401_still_exchanges_the_api_key_again_without_waiting(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(401, {})]
    async with worker(compute):
        pass
    assert clock.sleeps == []
    assert [s.authorization for s in pipeline.sockets] == ["Bearer minted-2"]


async def test_a_401_for_a_given_access_token_raises_at_once(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(401, {})]
    with pytest.raises(aiohttp.ClientResponseError) as refused:
        async with worker(compute, api_key=None, access_token="given"):
            pass
    assert refused.value.status == 401
    assert clock.sleeps == []


async def test_the_wait_is_logged_once_at_info_and_each_retry_at_debug(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, caplog: pytest.LogCaptureFixture):
    pipeline.upgrade_answers = [(503, {"Retry-After": "2"})] * 3
    with caplog.at_level(logging.DEBUG, logger="eyepop.requests"):
        async with worker(compute):
            pass
    waits = [r for r in caplog.records if "no capacity" in r.getMessage()]
    assert [r.levelno for r in waits] == [logging.INFO]
    assert "600 s" in waits[0].getMessage()
    retries = [r for r in caplog.records if "retrying in" in r.getMessage()]
    assert len(retries) == 3 and all(r.levelno == logging.DEBUG for r in retries)


async def test_a_job_waits_for_a_socket_to_be_opened_after_the_pipeline_closes_one(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    async with worker(compute) as endpoint:
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        job = await endpoint.load_from("https://images.example/a.jpg")
        with pytest.raises(PipelineSocketClosedException):
            while await job.predict() is not None:
                pass
        pipeline.upgrade_answers = [(503, {"Retry-After": "30"})]
        job = await endpoint.load_from("https://images.example/b.jpg")
        results = []
        while (result := await job.predict()) is not None:
            results.append(result)
    assert results == [prediction(0), prediction(1)]
    assert 30.0 in clock.sleeps


async def test_a_live_source_refused_overloaded_twice_raises_the_overloaded_exception(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.overloaded_live = 2
    async with worker(compute) as endpoint:
        job = await endpoint.load_from("rtsp://camera.example/stream")
        with pytest.raises(PipelineOverloadedException) as refused:
            await job.predict()
    assert refused.value.error == "overloaded"
    assert refused.value.waited_secs is None
    assert "overloaded" in str(refused.value)
