"""Pipeline mode: a refused socket (HTTP 503 or 429) is waited through, honouring Retry-After, and then raises."""
import asyncio
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
from eyepop.worker.pipeline_backend import PipelineBackend, _refusal_wait
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


@pytest.fixture
def no_jitter(monkeypatch: pytest.MonkeyPatch):
    """Every jitter takes its lowest value, so that a wait is the one the backoff or Retry-After names."""
    monkeypatch.setattr(pipeline_backend.random, "uniform", lambda low, high: low)


def worker(compute: FakeComputeAPI, **options) -> WorkerEndpoint:
    options.setdefault("api_key", "the api key")
    return EyePopSdk.async_worker(eyepop_url=compute.url, is_pipeline_mode=True, **options)


def test_the_wait_defaults_to_ten_minutes_and_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    assert Settings().pipeline_wait_timeout == 600.0
    monkeypatch.setenv("EYEPOP_PIPELINE_WAIT_TIMEOUT", "5")
    assert Settings().pipeline_wait_timeout == 5.0


async def test_the_upgrade_waits_the_retry_after_and_a_repeated_one_grows(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, no_jitter: None):
    pipeline.upgrade_answers = [(503, {"Retry-After": "3"}), (503, {"Retry-After": "3"}), (503, {"Retry-After": "3"})]
    async with worker(compute):
        pass
    assert clock.sleeps == [3.0, 4.5, 6.75]
    assert len(pipeline.sockets) == 1


async def test_a_429_is_waited_through_like_a_503(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, no_jitter: None):
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


def test_a_retry_after_gets_jitter_on_top_and_never_less_than_it_asked():
    waits = [_refusal_wait({"Retry-After": "30"}, 0) for _ in range(200)]
    assert all(30.0 <= wait <= 45.0 for wait in waits)
    assert len(set(waits)) > 1, "no jitter"


def test_a_repeated_retry_after_grows_to_ten_seconds_and_a_longer_one_is_kept(no_jitter: None):
    assert [_refusal_wait({"Retry-After": "1"}, n) for n in range(7)] == [
        1.0, 1.5, 2.25, 3.375, 5.0625, 7.59375, 10.0]
    assert _refusal_wait({"Retry-After": "1"}, 1000) == 10.0
    assert _refusal_wait({"Retry-After": "30"}, 5) == 30.0


@pytest.mark.parametrize("asked", ["0", "0.5", "1"])
def test_a_retry_after_below_a_second_is_a_second(asked: str):
    assert 1.0 <= _refusal_wait({"Retry-After": asked}, 0) <= 1.5


@pytest.mark.parametrize("asked", ["Wed, 21 Oct 2026 07:28:00 GMT", "-3", "inf", "-inf", "nan", ""])
def test_an_unusable_retry_after_falls_back_to_the_backoff(asked: str):
    assert all(0.5 <= _refusal_wait({"Retry-After": asked}, 0) <= 1.0 for _ in range(50))


def test_the_backoff_does_not_overflow_after_many_refusals():
    assert all(5.0 <= _refusal_wait({}, 10 ** 6) <= 10.0 for _ in range(50))
    assert all(5.0 <= _refusal_wait({}, 1030) <= 10.0 for _ in range(50))


async def test_a_socket_refused_for_the_whole_wait_raises_overloaded(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 20.0)
    pipeline.upgrade_answers = [(503, {"Retry-After": "8"})] * 10
    with pytest.raises(PipelineOverloadedException) as refused:
        async with worker(compute):
            pass
    # 8 s, then 10 s (8 grown, capped at 10); the next wait, 10 s, would pass the 20 s limit.
    assert clock.sleeps == [8.0, 10.0]
    assert "stayed at capacity for 18 seconds" in str(refused.value)
    assert "pipeline_wait_timeout (20 s)" in str(refused.value)
    assert refused.value.waited_secs == 18.0
    assert refused.value.error == "overloaded"
    assert isinstance(refused.value, PipelineCommandException)
    assert isinstance(refused.value.__cause__, aiohttp.ClientResponseError)
    assert len(pipeline.upgrade_answers) == 10 - 3, "three tries, and none after the last Retry-After"
    assert pipeline.sockets == []


async def test_a_retry_after_past_the_limit_raises_at_once_without_another_try(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(503, {"Retry-After": "1e9"})] * 3
    with pytest.raises(PipelineOverloadedException, match="leaves no time to wait"):
        async with worker(compute):
            pass
    assert clock.sleeps == []
    assert len(pipeline.upgrade_answers) == 2, "one try"


@pytest.mark.parametrize("asked", ["nan", "inf"])
async def test_a_retry_after_that_is_not_a_number_to_wait_is_the_backoff(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, asked: str):
    pipeline.upgrade_answers = [(503, {"Retry-After": asked})]
    async with worker(compute):
        pass
    assert len(clock.sleeps) == 1 and 0.5 <= clock.sleeps[0] <= 1.0


async def test_a_wait_of_zero_does_not_wait(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 0.0)
    pipeline.upgrade_answers = [(503, {"Retry-After": "1"})]
    with pytest.raises(PipelineOverloadedException, match="pipeline_wait_timeout \\(0 s\\) leaves no time"):
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


async def test_the_first_wait_is_a_warning_and_each_retry_is_debug(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, caplog: pytest.LogCaptureFixture):
    pipeline.upgrade_answers = [(503, {"Retry-After": "2"})] * 3
    with caplog.at_level(logging.DEBUG, logger="eyepop.requests"):
        async with worker(compute):
            pass
    waits = [r for r in caplog.records if "no capacity" in r.getMessage()]
    assert [r.levelno for r in waits] == [logging.WARNING]
    assert "503 (Retry-After 2)" in waits[0].getMessage()
    assert "600 s" in waits[0].getMessage()
    retries = [r for r in caplog.records if "retrying in" in r.getMessage()]
    assert len(retries) == 3 and all(r.levelno == logging.DEBUG for r in retries)


async def test_the_wait_shows_at_the_default_level(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, caplog: pytest.LogCaptureFixture):
    pipeline.upgrade_answers = [(503, {})] * 2
    with caplog.at_level(logging.WARNING, logger="eyepop.requests"):
        async with worker(compute):
            pass
    assert len([r for r in caplog.records if "no capacity" in r.getMessage()]) == 1


async def test_a_401_exchanges_the_api_key_once_and_not_again_on_each_later_refusal(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(401, {}), (503, {"Retry-After": "1"}), (503, {"Retry-After": "1"}),
                                (503, {"Retry-After": "1"})]
    async with worker(compute):
        pass
    assert len(compute.exchanges) == 2, "the first token, and the one after the 401"
    assert len(clock.sleeps) == 3
    assert [s.authorization for s in pipeline.sockets] == ["Bearer minted-2"]


async def test_a_second_401_raises(compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    pipeline.upgrade_answers = [(401, {}), (401, {})]
    with pytest.raises(aiohttp.ClientResponseError) as refused:
        async with worker(compute):
            pass
    assert refused.value.status == 401
    assert len(compute.exchanges) == 2


async def test_a_job_waits_for_a_socket_to_be_opened_after_the_pipeline_closes_one(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, no_jitter: None):
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


async def test_jobs_queued_behind_a_wait_that_gave_up_fail_at_once_not_after_a_wait_each(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 100.0)
    async with worker(compute) as endpoint:
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        with pytest.raises(PipelineSocketClosedException):
            await (await endpoint.load_from("https://images.example/first.jpg")).predict()
        pipeline.upgrade_answers = [(503, {"Retry-After": "10"})] * 1000
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(3)]
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(job.predict() for job in jobs), return_exceptions=True), 5)
        tries = 1000 - len(pipeline.upgrade_answers)
    assert all(isinstance(o, PipelineOverloadedException) for o in outcomes), outcomes
    assert sum(clock.sleeps) <= 100.0, "one wait for the three jobs, not one each"
    # Only the first job asked: the tries are its own, 10 s apart.
    assert tries == len(clock.sleeps) + 1
    assert outcomes[1].waited_secs == outcomes[0].waited_secs == outcomes[2].waited_secs


async def test_a_send_after_the_memory_of_a_given_up_wait_waits_again(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 30.0)
    async with worker(compute) as endpoint:
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        with pytest.raises(PipelineSocketClosedException):
            await (await endpoint.load_from("https://images.example/first.jpg")).predict()
        pipeline.upgrade_answers = [(503, {"Retry-After": "40"})]
        with pytest.raises(PipelineOverloadedException):
            await (await endpoint.load_from("https://images.example/a.jpg")).predict()
        clock.now += 60
        assert len(pipeline.upgrade_answers) == 0
        assert await (await endpoint.load_from("https://images.example/b.jpg")).predict() == prediction(0)
    assert len(pipeline.sockets) == 2


async def test_disconnect_ends_a_wait_at_once(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, monkeypatch: pytest.MonkeyPatch):
    async def never(seconds: float) -> None:
        await asyncio.Event().wait()

    async with worker(compute) as endpoint:
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        with pytest.raises(PipelineSocketClosedException):
            await (await endpoint.load_from("https://images.example/first.jpg")).predict()
        monkeypatch.setattr(pipeline_backend, "_sleep", never)
        pipeline.upgrade_answers = [(503, {"Retry-After": "1"})] * 1000
        job = await endpoint.load_from("https://images.example/a.jpg")
        for _ in range(200):
            if len(pipeline.upgrade_answers) < 1000:
                break
            await asyncio.sleep(0.01)
        assert len(pipeline.upgrade_answers) == 999, "the job is waiting for capacity"
        await asyncio.wait_for(endpoint.disconnect(), 5)
        with pytest.raises(PipelineSocketClosedException, match="disconnecting"):
            await job.predict()
    assert len(pipeline.upgrade_answers) == 999, "no try after the disconnect"


async def test_a_successful_upgrade_clears_a_given_up_wait(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    async with worker(compute) as endpoint:
        backend = endpoint.backend
        assert isinstance(backend, PipelineBackend)
        backend._gave_up = (clock.now, PipelineOverloadedException("earlier", waited_secs=1.0))
        ws = await backend._upgrade(await backend._socket_url(), patient=False)
        await ws.close()
        assert backend._gave_up is None


async def test_the_reconnect_and_a_later_refusal_are_not_failed_by_a_given_up_wait(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 30.0)
    async with worker(compute) as endpoint:
        backend = endpoint.backend
        assert isinstance(backend, PipelineBackend)
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        with pytest.raises(PipelineSocketClosedException):
            await (await endpoint.load_from("https://images.example/first.jpg")).predict()
        pipeline.upgrade_answers = [(503, {"Retry-After": "40"})]
        with pytest.raises(PipelineOverloadedException):
            await (await endpoint.load_from("https://images.example/a.jpg")).predict()
        assert backend._gave_up is not None
        # No time has passed, so the record is still inside its window.
        await backend._reconnect_at_once()
        assert len(pipeline.sockets) == 2 and backend._gave_up is None
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        with pytest.raises(PipelineSocketClosedException):
            await (await endpoint.load_from("https://images.example/b.jpg")).predict()
        pipeline.upgrade_answers = [(503, {"Retry-After": "3"})]
        assert await (await endpoint.load_from("https://images.example/c.jpg")).predict() == prediction(0)
    assert clock.sleeps == [3.0], "the refusal was waited through, not failed at once"
    assert len(pipeline.sockets) == 3


async def give_up(endpoint: WorkerEndpoint, pipeline: FakePipelineAPI, retry_after: str = "40") -> PipelineOverloadedException:
    """Close the endpoint's socket and refuse the next upgrade for longer than the limit, so a job gives up."""
    pipeline.closes = {1: (1011, "pipeline_stopped")}
    with pytest.raises(PipelineSocketClosedException):
        await (await endpoint.load_from("https://images.example/first.jpg")).predict()
    pipeline.upgrade_answers = [(503, {"Retry-After": retry_after})] * 5
    with pytest.raises(PipelineOverloadedException) as refused:
        await (await endpoint.load_from("https://images.example/gave-up.jpg")).predict()
    return refused.value


async def test_a_long_queue_of_jobs_a_round_trip_apart_all_fail_at_once(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 10.0)
    original = PipelineBackend._open_socket

    async def two_seconds_each(self: PipelineBackend, patient: bool):
        clock.now += 2.0  # what each send's own round trip takes
        return await original(self, patient)

    async with worker(compute) as endpoint:
        pipeline.closes = {1: (1011, "pipeline_stopped")}
        with pytest.raises(PipelineSocketClosedException):
            await (await endpoint.load_from("https://images.example/first.jpg")).predict()
        monkeypatch.setattr(PipelineBackend, "_open_socket", two_seconds_each)
        pipeline.upgrade_answers = [(503, {"Retry-After": "1"})] * 1000
        started = clock.now
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(5)]
        outcomes = await asyncio.wait_for(
            asyncio.gather(*(job.predict() for job in jobs), return_exceptions=True), 5)
        tries = 1000 - len(pipeline.upgrade_answers)
    assert all(isinstance(o, PipelineOverloadedException) for o in outcomes), outcomes
    assert sum(clock.sleeps) <= 10.0, "one wait for the five jobs"
    assert tries == len(clock.sleeps) + 1, "only the first job asked"
    assert clock.now - started > 5.0, "the queue took longer than the old fixed window"


async def test_a_job_after_thirty_quiet_seconds_waits_afresh_and_one_before_that_does_not(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 30.0)
    async with worker(compute) as endpoint:
        await give_up(endpoint, pipeline)
        for _ in range(3):
            clock.now += 29.0  # each fail-fast renews the record, so the gaps add up to far more than 30 s
            with pytest.raises(PipelineOverloadedException):
                await (await endpoint.load_from("https://images.example/queued.jpg")).predict()
        assert len(pipeline.upgrade_answers) == 4, "no try went out for the three"
        clock.now += 31.0
        pipeline.upgrade_answers = [(503, {"Retry-After": "3"})]
        assert await (await endpoint.load_from("https://images.example/b.jpg")).predict() == prediction(0)
    assert clock.sleeps == [3.0]


async def test_connect_and_disconnect_each_clear_a_given_up_wait(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    async with worker(compute) as endpoint:
        backend = endpoint.backend
        assert isinstance(backend, PipelineBackend)
        earlier = PipelineOverloadedException("earlier", waited_secs=1.0)
        backend._gave_up = (clock.now, earlier)
        await backend.connect()
        assert backend._gave_up is None
        backend._gave_up = (clock.now, earlier)
        await backend.disconnect()
        assert backend._gave_up is None


async def test_a_fail_fast_reports_the_give_ups_own_wait_not_the_time_since(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock, monkeypatch: pytest.MonkeyPatch,
        no_jitter: None):
    monkeypatch.setattr(settings, "pipeline_wait_timeout", 20.0)
    async with worker(compute) as endpoint:
        first = await give_up(endpoint, pipeline, retry_after="8")  # waits 8 s, then 10 s would pass the limit
        assert first.waited_secs == 18.0
        pipeline.upgrade_answers = [(503, {"Retry-After": "8"})] * 5
        clock.now += 25.0
        with pytest.raises(PipelineOverloadedException) as again:
            await (await endpoint.load_from("https://images.example/a.jpg")).predict()
    assert again.value.waited_secs == 18.0
    assert str(again.value) == str(first)
    assert "3610" not in str(again.value)


async def test_a_fail_fast_after_a_give_up_that_did_not_wait_reports_no_wait_and_a_late_job_its_own(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, clock: Clock):
    async with worker(compute) as endpoint:
        first = await give_up(endpoint, pipeline, retry_after="1e9")
        assert first.waited_secs is not None and first.waited_secs < 0.5
        clock.now += 20.0
        with pytest.raises(PipelineOverloadedException) as again:
            await (await endpoint.load_from("https://images.example/a.jpg")).predict()
        assert again.value.waited_secs == first.waited_secs
        assert "leaves no time to wait" in str(again.value)
        # An hour later the record has lapsed: this job's own try is refused, and it reports its own wait.
        clock.now += 3600.0
        with pytest.raises(PipelineOverloadedException) as late:
            await (await endpoint.load_from("https://images.example/b.jpg")).predict()
        assert late.value.waited_secs is not None and late.value.waited_secs < 0.5
        assert late.value is not again.value
