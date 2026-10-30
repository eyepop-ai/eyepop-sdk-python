"""Pipeline mode (SDK-24): WorkerEndpoint's jobs on the pipeline API's socket, against the fakes."""
import asyncio
import contextlib
import io
import threading
import warnings
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from eyepop import EyePopSdk
from eyepop.exceptions import (
    PipelineCommandException,
    PipelineSocketClosedException,
    PipelineSourceException,
    PopConfigurationException,
)
from eyepop.settings import settings
from eyepop.worker.pipeline_backend import PipelineBackend
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_jobs import WorkerJob
from eyepop.worker.worker_types import InferenceComponent, Pop
from tests.worker.pipeline_fakes import FakeComputeAPI, FakePipelineAPI, prediction

PERSON = Pop(components=[InferenceComponent(ability="eyepop.person:latest")])
VEHICLE = Pop(components=[InferenceComponent(ability="eyepop.vehicle:latest")])


@pytest.fixture(autouse=True)
def no_environment(monkeypatch: pytest.MonkeyPatch):
    for name in ("EYEPOP_URL", "EYEPOP_API_KEY", "EYEPOP_SECRET_KEY", "EYEPOP_ACCESS_TOKEN", "EYEPOP_POP_ID",
                 "EYEPOP_SESSION_UUID", "EYEPOP_PIPELINE_MODE", "EYEPOP_LOCAL_MODE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def one_socket(monkeypatch: pytest.MonkeyPatch):
    """An endpoint that opens one socket, for what a socket does with the sources it holds."""
    monkeypatch.setattr(settings, "pipeline_sockets", 1)


@pytest.fixture
async def pipeline() -> AsyncIterator[FakePipelineAPI]:
    async with FakePipelineAPI() as fake:
        yield fake


@pytest.fixture
async def compute(pipeline: FakePipelineAPI) -> AsyncIterator[FakeComputeAPI]:
    async with FakeComputeAPI(pipeline.url) as fake:
        yield fake


def in_pipeline_mode(endpoint: object) -> bool:
    return isinstance(endpoint, WorkerEndpoint) and isinstance(endpoint.backend, PipelineBackend)


def worker(compute: FakeComputeAPI, **options) -> WorkerEndpoint:
    options.setdefault("api_key", "the api key")
    endpoint = EyePopSdk.async_worker(eyepop_url=compute.url, is_pipeline_mode=True, **options)
    assert in_pipeline_mode(endpoint)
    return endpoint


def pipeline_backend(endpoint: WorkerEndpoint) -> PipelineBackend:
    assert isinstance(endpoint.backend, PipelineBackend)
    return endpoint.backend


async def predictions(job: WorkerJob) -> list[dict]:
    results = []
    while (result := await job.predict()) is not None:
        results.append(result)
    return results


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "image.jpg"
    path.write_bytes(b"jpeg bytes")
    return path


# 1. Selection


def test_pipeline_mode_is_selected_by_the_argument_or_the_environment(monkeypatch: pytest.MonkeyPatch):
    assert in_pipeline_mode(EyePopSdk.async_worker(is_pipeline_mode=True, access_token="t"))
    monkeypatch.setenv("EYEPOP_PIPELINE_MODE", "true")
    assert in_pipeline_mode(EyePopSdk.async_worker(access_token="t"))
    assert not in_pipeline_mode(EyePopSdk.async_worker(access_token="t", is_pipeline_mode=False))


@pytest.mark.parametrize("option", [
    {"pop_id": "a-named-pop"}, {"session_uuid": "a-session"}, {"pipeline_image": "an-image"},
    {"pipeline_version": "4.8.0"}, {"session_name": "a-name"}, {"dataset_uuid": "a-dataset"},
    {"secret_key": "a secret key"}, {"job_queue_length": 257},
])
def test_pipeline_mode_refuses_what_only_a_worker_session_has(option: dict):
    with pytest.raises(ValueError):
        EyePopSdk.async_worker(is_pipeline_mode=True, api_key="k", **option)


def test_pipeline_mode_refuses_a_session_named_by_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EYEPOP_SESSION_UUID", "a-session")
    with pytest.raises(ValueError, match="session_uuid"):
        EyePopSdk.async_worker(is_pipeline_mode=True, api_key="k")


def test_pipeline_mode_holds_up_to_256_jobs_by_default():
    assert EyePopSdk.async_worker(is_pipeline_mode=True, api_key="k").sem._value == 256
    assert EyePopSdk.async_worker(is_pipeline_mode=True, api_key="k", job_queue_length=8).sem._value == 8


async def test_the_pipeline_api_url_comes_from_the_compute_apis_configs(compute: FakeComputeAPI,
                                                                        pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        assert await endpoint.session() == {"pipeline_api_url": pipeline.url}
    assert len(pipeline.sockets) == 1
    assert compute.config_authorizations == ["Bearer minted-1"]


async def test_a_compute_api_without_a_pipeline_api_fails_the_connect_naming_the_key(pipeline: FakePipelineAPI):
    async with FakeComputeAPI(None) as compute:
        with pytest.raises(PopConfigurationException, match="pipeline_api_url"):
            await worker(compute).connect()


async def test_plain_http_is_refused_for_a_pipeline_api_off_loopback():
    async with FakeComputeAPI("http://pipeline.example") as compute:
        with pytest.raises(ValueError, match="https"):
            await worker(compute).connect()


async def test_local_mode_takes_eyepop_url_as_the_pipeline_api(pipeline: FakePipelineAPI):
    endpoint = EyePopSdk.async_worker(is_pipeline_mode=True, is_local_mode=True, eyepop_url=pipeline.url)
    async with endpoint:
        assert await predictions(await endpoint.load_from("https://images.example/a.jpg")) == [prediction(0), prediction(1)]
    assert pipeline.sockets[0].authorization is None


# 2. Options the socket does not take yet


async def test_an_option_not_sent_warns_once_per_endpoint_naming_awsu_317(compute: FakeComputeAPI,
                                                                         pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                await predictions(await endpoint.load_from("https://images.example/a.jpg", media_cache_seconds=10))
    assert [str(w.message) for w in caught] == [
        "media_cache_seconds is not sent to the pipeline API and has no effect in pipeline mode (AWSU-317)"]
    assert all("mediaCacheSeconds" not in source for source in pipeline.sockets[0].sources)


# 3. Session-era methods


async def test_session_era_methods_raise_and_get_pop_returns_the_last_pop(compute: FakeComputeAPI):
    async with worker(compute, pop=PERSON) as endpoint:
        await endpoint.set_pop(VEHICLE)
        assert await endpoint.get_pop() == VEHICLE
        with pytest.raises(NotImplementedError):
            await endpoint.pipeline_get("pop")
        with pytest.raises(NotImplementedError):
            await endpoint.dev_mode_pipeline_base_url()


# 4. Lifecycle


async def test_connect_opens_the_socket_and_sets_the_pop(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute, pop=PERSON):
        pass
    assert pipeline.sockets[0].protocols == ["eyepop.v1"]
    assert pipeline.sockets[0].pops == [PERSON.model_dump(mode="json")]


async def test_set_pop_sends_set_pop_and_a_refused_pop_raises(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        await endpoint.set_pop(PERSON)
        pipeline.pop_error = "invalid_request"
        with pytest.raises(PopConfigurationException, match="invalid_request"):
            await endpoint.set_pop(VEHICLE)
        assert await endpoint.get_pop() == PERSON, "the refused pop is not the one a reconnect sets"
    assert len(pipeline.sockets[0].pops) == 2


async def test_a_job_cancelled_before_its_source_went_out_ends(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        job = await endpoint.load_from("rtsp://camera.example/stream")
        await job.cancel()
        await asyncio.wait_for(asyncio.gather(*endpoint.tasks), 5)


async def test_after_an_idle_close_the_next_job_reconnects_and_sets_the_pop_again(compute: FakeComputeAPI,
                                                                                 pipeline: FakePipelineAPI):
    pipeline.idle_after_done = True
    async with worker(compute, pop=PERSON) as endpoint:
        await predictions(await endpoint.load_from("https://images.example/a.jpg"))
        await asyncio.sleep(0.05)
        assert await predictions(await endpoint.load_from("https://images.example/b.jpg")) == [prediction(0), prediction(1)]
    assert len(pipeline.sockets) == 2
    assert pipeline.sockets[1].pops == [PERSON.model_dump(mode="json")]


async def test_credential_expired_reconnects_at_once_with_a_new_token(compute: FakeComputeAPI,
                                                                      pipeline: FakePipelineAPI):
    pipeline.closes = {1: (1008, "credential_expired")}
    async with worker(compute, pop=PERSON) as endpoint:
        with pytest.raises(PipelineSocketClosedException):
            await predictions(await endpoint.load_from("https://images.example/a.jpg"))
        for _ in range(100):
            if len(pipeline.sockets) == 2 and pipeline.sockets[1].pops:
                break
            await asyncio.sleep(0.01)
        assert [s.authorization for s in pipeline.sockets] == ["Bearer minted-1", "Bearer minted-2"]
        assert pipeline.sockets[1].pops == [PERSON.model_dump(mode="json")]
        assert pipeline.sockets[1].sources == [], "the job in flight was not sent again"


async def test_a_job_in_flight_when_the_pipeline_stops_fails_with_the_close(compute: FakeComputeAPI,
                                                                            pipeline: FakePipelineAPI):
    pipeline.closes = {1: (1011, "pipeline_stopped")}
    async with worker(compute) as endpoint:
        job = await endpoint.load_from("https://images.example/a.jpg")
        with pytest.raises(PipelineSocketClosedException) as closed:
            await predictions(job)
        assert (closed.value.code, closed.value.reason) == (1011, "pipeline_stopped")
        assert await predictions(await endpoint.load_from("https://images.example/b.jpg")) == [prediction(0), prediction(1)]
    assert len(pipeline.sockets) == 2, "the next job opened a new socket; the failed one was not sent again"
    assert [s["url"] for s in pipeline.sockets[1].sources] == ["https://images.example/b.jpg"]


async def test_the_upgrade_waits_through_503_with_retry_after(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.upgrade_answers = [(503, {"Retry-After": "0"})]
    async with worker(compute):
        pass
    assert len(pipeline.sockets) == 1


async def test_a_live_job_refused_overloaded_goes_once_more_on_a_new_socket(compute: FakeComputeAPI,
                                                                            pipeline: FakePipelineAPI):
    pipeline.overloaded_live = 1
    async with worker(compute) as endpoint:
        job = await endpoint.load_from("rtsp://camera.example/stream")
        assert await job.predict() == prediction(0)
        await job.cancel()
    assert len(pipeline.sockets) == 2
    assert pipeline.sockets[1].sources[0]["url"] == "rtsp://camera.example/stream"


async def test_a_live_job_refused_overloaded_twice_raises(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.overloaded_live = 2
    async with worker(compute) as endpoint:
        with pytest.raises(PipelineCommandException, match="overloaded"):
            await predictions(await endpoint.load_from("rtsp://camera.example/stream"))


async def test_closes_one_after_another_wait_longer_before_each_reconnect(compute: FakeComputeAPI,
                                                                          pipeline: FakePipelineAPI):
    # Every socket is closed `1001 shutting_down` as it opens: the first reconnect is at once, the
    # next waits a second, the one after two
    pipeline.close_on_open = (1001, "shutting_down")
    async with worker(compute):
        await asyncio.sleep(0.5)
        assert len(pipeline.sockets) == 2
        await asyncio.sleep(1)
        assert len(pipeline.sockets) == 3
        await asyncio.sleep(1)
        assert len(pipeline.sockets) == 3


async def test_a_disconnect_timeout_closes_the_socket_under_a_live_job(compute: FakeComputeAPI,
                                                                       pipeline: FakePipelineAPI):
    endpoint = worker(compute)
    await endpoint.connect()
    job = await endpoint.load_from("rtsp://camera.example/stream")
    assert await job.predict() == prediction(0)

    await endpoint.disconnect(timeout=0.2)

    backend = endpoint.backend
    assert isinstance(backend, PipelineBackend)
    assert backend._sockets == [], "the socket closed although the live job never ended on its own"
    with contextlib.suppress(PipelineSocketClosedException):
        while await asyncio.wait_for(job.predict(), 5) is not None:
            pass
    # A timed out disconnect skips the endpoint's own cleanup, as it always has.
    if endpoint.client_session is not None:
        await endpoint.client_session.close()


async def test_a_job_cancelled_before_its_turn_never_sends_its_set_source(compute: FakeComputeAPI,
                                                                          pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        # the setPop holds the line until it is answered; the job submitted after it waits its turn
        pipeline.pop_delay = 0.2
        changing = asyncio.create_task(endpoint.set_pop(PERSON))
        await asyncio.sleep(0)
        job = await endpoint.load_from("https://images.example/never.jpg")
        await asyncio.sleep(0.05)  # the job runs and waits for its turn behind the setPop
        await job.cancel()
        await changing
        assert await job.predict() is None
    assert pipeline.sockets[0].sources == []


async def test_a_cancelled_live_job_stops_its_source_and_the_next_job_plays(compute: FakeComputeAPI,
                                                                            pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        live = await endpoint.load_from("rtsp://camera.example/stream")
        assert await live.predict() == prediction(0)
        await live.cancel()
        # the live source no longer holds the pipeline's queue
        image = await endpoint.load_from("https://images.example/a.jpg")
        assert await asyncio.wait_for(predictions(image), 5) == [prediction(0), prediction(1)]
    socket = pipeline.sockets[0]
    assert {"type": "stopSource", "id": socket.sources[0]["id"]} in socket.commands


async def test_cancelling_against_a_pipeline_api_without_stop_source_still_ends_the_job(compute: FakeComputeAPI,
                                                                                        pipeline: FakePipelineAPI):
    pipeline.knows_stop_source = False
    async with worker(compute) as endpoint:
        live = await endpoint.load_from("rtsp://camera.example/stream")
        assert await live.predict() == prediction(0)
        await live.cancel()
        assert await live.predict() is None


async def test_a_source_that_ends_with_an_error_raises(compute: FakeComputeAPI):
    async with worker(compute) as endpoint:
        with pytest.raises(PipelineSourceException, match="bad media"):
            await predictions(await endpoint.load_from("https://images.example/bad.jpg"))


# 5. Uploads


async def test_a_small_file_goes_combined(compute: FakeComputeAPI, pipeline: FakePipelineAPI, image: Path):
    async with worker(compute) as endpoint:
        assert await predictions(await endpoint.upload(str(image))) == [prediction(0), prediction(1)]
    socket = pipeline.sockets[0]
    source = socket.sources[0]
    assert source["upload"] == {"contentType": "image/jpeg"}
    assert socket.combined == [source["id"]]
    assert socket.media[source["id"]] == {None: b"jpeg bytes"}


async def test_a_file_over_the_combined_limit_goes_in_chunks_within_credit(compute: FakeComputeAPI,
                                                                         pipeline: FakePipelineAPI, image: Path,
                                                                         monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "combined_upload_max_bytes", 4)
    pipeline.window = 3
    async with worker(compute) as endpoint:
        await predictions(await endpoint.upload(str(image)))
    socket = pipeline.sockets[0]
    source = socket.sources[0]
    assert socket.combined == []
    assert socket.media[source["id"]] == {None: b"jpeg bytes"}
    assert {"type": "uploaded", "id": source["id"]} in socket.commands


async def test_a_stream_and_a_live_upload_go_in_chunks(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async def pieces():
        yield b"video "
        yield b"bytes"

    async with worker(compute) as endpoint:
        await predictions(await endpoint.upload_stream(io.BytesIO(b"mp4 bytes"), "video/mp4"))
        live = await endpoint.upload_stream(pieces(), "video/mp4", is_live=True)
        assert await live.predict() == prediction(0)
        await live.cancel()
    socket = pipeline.sockets[0]
    first, second = socket.sources
    assert socket.combined == []
    assert socket.media[first["id"]] == {None: b"mp4 bytes"}
    assert socket.media[second["id"]] == {None: b"video bytes"}
    assert second["live"] is True


async def test_a_group_of_files_goes_combined_with_each_members_bytes(compute: FakeComputeAPI,
                                                                      pipeline: FakePipelineAPI, tmp_path: Path):
    (tmp_path / "a.png").write_bytes(b"one")
    (tmp_path / "b.jpg").write_bytes(b"two2")
    async with worker(compute) as endpoint:
        await predictions(await endpoint.upload_group([str(tmp_path / "a.png"), str(tmp_path / "b.jpg")]))
    socket = pipeline.sockets[0]
    source = socket.sources[0]
    assert source["group"] == [{"upload": {"contentType": "image/png", "bytes": 3}},
                               {"upload": {"contentType": "image/jpeg", "bytes": 4}}]
    assert socket.media[source["id"]] == {0: b"one", 1: b"two2"}


async def test_a_group_of_streams_uploads_each_member_in_chunks(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        await predictions(await endpoint.upload_stream_group([io.BytesIO(b"first"), io.BytesIO(b"second")],
                                                             mime_types=["image/png", "image/jpeg"]))
    socket = pipeline.sockets[0]
    source = socket.sources[0]
    assert source["group"] == [{"upload": {"contentType": "image/png"}}, {"upload": {"contentType": "image/jpeg"}}]
    assert socket.media[source["id"]] == {0: b"first", 1: b"second"}
    assert {"type": "uploaded", "id": source["id"], "member": 1} in socket.commands


async def test_load_from_group_and_load_asset_name_their_source(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        await predictions(await endpoint.load_from_group(["https://images.example/a.jpg", "https://images.example/b.jpg"]))
        await predictions(await endpoint.load_asset("an-asset"))
    group, asset = pipeline.sockets[0].sources
    assert group["group"] == [{"url": "https://images.example/a.jpg"}, {"url": "https://images.example/b.jpg"}]
    assert asset["assetUuid"] == "an-asset"
    assert "mode" not in group and "version" not in group, "queue is the default mode; the version is the API's"


# 6. Concurrency


@pytest.mark.usefixtures("one_socket")
async def test_jobs_go_out_in_submission_order_on_one_socket(compute: FakeComputeAPI, pipeline: FakePipelineAPI,
                                                             image: Path):
    urls = [f"https://images.example/{n}.jpg" for n in range(5)]
    async with worker(compute) as endpoint:
        jobs = [await endpoint.upload(str(image))] + [await endpoint.load_from(url) for url in urls]
        results = await asyncio.gather(*(predictions(job) for job in jobs))
    assert all(r == [prediction(0), prediction(1)] for r in results)
    assert len(pipeline.sockets) == 1
    assert [s.get("url") for s in pipeline.sockets[0].sources] == [None, *urls]


@pytest.mark.usefixtures("one_socket")
async def test_queue_full_with_no_other_source_open_raises(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.refusals = ["queue_full"]
    async with worker(compute) as endpoint:
        with pytest.raises(PipelineCommandException, match="queue_full"):
            await predictions(await endpoint.load_from("https://images.example/a.jpg"))


@pytest.mark.usefixtures("one_socket")
async def test_sources_refused_with_none_taken_raise_instead_of_sending_each_other_again(compute: FakeComputeAPI,
                                                                                        pipeline: FakePipelineAPI):
    pipeline.queue_full_at = 0
    async with worker(compute) as endpoint:
        jobs = [await endpoint.load_from(f"https://images.example/{name}.jpg") for name in "ab"]
        refused = await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs), return_exceptions=True), 5)
    assert [e.error for e in refused] == ["queue_full", "queue_full"]
    assert pipeline.queue_full_refused == 2


@pytest.mark.usefixtures("one_socket")
async def test_a_source_refused_queue_full_goes_again_once_another_ended(compute: FakeComputeAPI,
                                                                          pipeline: FakePipelineAPI):
    pipeline.queue_full_at = 2
    pipeline.source_delay = 0.02
    urls = [f"https://images.example/{n}.jpg" for n in range(30)]
    async with worker(compute) as endpoint:
        jobs = [await endpoint.load_from(url) for url in urls]
        results = await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 10)
    assert all(r == [prediction(0), prediction(1)] for r in results)
    assert pipeline.queue_full_refused > 0
    # While the socket is full each end lets one send go, so no source is refused twice.
    assert pipeline.queue_full_refused <= len(urls)
    assert len(pipeline.sockets) == 1
    sent = [s["url"] for s in pipeline.sockets[0].sources]
    assert sorted(set(sent)) == sorted(urls) and len(sent) == len(urls) + pipeline.queue_full_refused


@pytest.mark.usefixtures("one_socket")
async def test_the_end_of_a_cancelled_source_lets_a_source_refused_queue_full_go(compute: FakeComputeAPI,
                                                                                pipeline: FakePipelineAPI):
    pipeline.queue_full_at = 1
    pipeline.source_delay = 0.3
    async with worker(compute) as endpoint:
        slow = await endpoint.load_from("https://images.example/slow.jpg")
        image = await endpoint.load_from("https://images.example/a.jpg")
        await asyncio.sleep(0.05)  # refused queue_full, it waits for room
        assert pipeline.queue_full_refused == 1
        await slow.cancel()
        assert await asyncio.wait_for(predictions(image), 5) == [prediction(0), prediction(1)]


@pytest.mark.usefixtures("one_socket")
async def test_a_job_cancelled_while_it_waits_for_room_ends_at_once(compute: FakeComputeAPI,
                                                                    pipeline: FakePipelineAPI):
    pipeline.queue_full_at = 1
    pipeline.source_delay = 0.3
    async with worker(compute) as endpoint:
        slow = await endpoint.load_from("https://images.example/slow.jpg")
        image = await endpoint.load_from("https://images.example/a.jpg")
        await asyncio.sleep(0.05)  # refused queue_full, it waits for room behind the slow source
        await image.cancel()
        for _ in range(100):
            if len(endpoint.tasks) == 1:
                break
            await asyncio.sleep(0.01)
        assert len(endpoint.tasks) == 1  # only the slow job runs on
        assert await predictions(slow) == [prediction(0), prediction(1)]
    assert pipeline.queue_full_refused == 1
    assert [s.get("url") for s in pipeline.sockets[0].sources] == ["https://images.example/slow.jpg",
                                                                   "https://images.example/a.jpg"]


@pytest.mark.usefixtures("one_socket")
async def test_an_upload_from_a_stream_refused_queue_full_raises(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.queue_full_at = 1
    pipeline.source_delay = 0.3
    async with worker(compute) as endpoint:
        slow = await endpoint.load_from("https://images.example/slow.jpg")
        # Its first bytes were read for the refused send; it cannot be sent again whole.
        job = await endpoint.upload_stream(io.BytesIO(b"jpeg bytes"), "image/jpeg")
        with pytest.raises(PipelineCommandException, match="queue_full"):
            await asyncio.wait_for(predictions(job), 5)
        await predictions(slow)


async def test_a_batch_spreads_over_pipeline_sockets_sockets_each_with_the_pop(compute: FakeComputeAPI,
                                                                              pipeline: FakePipelineAPI):
    pipeline.source_delay = 0.2
    urls = [f"https://images.example/{n}.jpg" for n in range(12)]
    async with worker(compute, pop=PERSON) as endpoint:
        jobs = [await endpoint.load_from(url) for url in urls]
        results = await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 5)
    assert all(r == [prediction(0), prediction(1)] for r in results)
    assert len(pipeline.sockets) == settings.pipeline_sockets == 4
    assert all(socket.pops == [PERSON.model_dump(mode="json")] for socket in pipeline.sockets)
    assert all(socket.sources for socket in pipeline.sockets)
    assert sorted(s["url"] for socket in pipeline.sockets for s in socket.sources) == sorted(urls)


async def test_jobs_one_after_another_stay_on_one_socket(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute) as endpoint:
        for n in range(3):
            assert await predictions(await endpoint.load_from(f"https://images.example/{n}.jpg")) == [
                prediction(0), prediction(1)]
    assert len(pipeline.sockets) == 1


async def test_set_pop_reaches_every_open_socket(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.source_delay = 0.2
    async with worker(compute, pop=PERSON) as endpoint:
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(4)]
        await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 5)
        await endpoint.set_pop(VEHICLE)
    assert len(pipeline.sockets) == 4
    assert all(socket.pops == [PERSON.model_dump(mode="json"), VEHICLE.model_dump(mode="json")]
               for socket in pipeline.sockets)


async def test_another_socket_the_pipeline_api_refuses_does_not_hold_up_the_job(compute: FakeComputeAPI,
                                                                               pipeline: FakePipelineAPI):
    pipeline.source_delay = 0.2
    async with worker(compute) as endpoint:
        pipeline.upgrade_answers = [(503, {"Retry-After": "5"})]
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(2)]
        results = await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 2)
    assert all(r == [prediction(0), prediction(1)] for r in results)
    assert len(pipeline.sockets) == 1, "the second job went on the open socket instead of waiting 5 s"


async def test_after_the_pipeline_api_refused_another_socket_the_next_jobs_do_not_ask_again_at_once(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.source_delay = 0.05
    async with worker(compute) as endpoint:
        pipeline.upgrade_answers = [(503, {}), (503, {}), (503, {})]
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(8)]
        results = await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 5)
    assert all(r == [prediction(0), prediction(1)] for r in results)
    assert len(pipeline.upgrade_answers) == 2
    assert len(pipeline.sockets) == 1


async def test_a_camera_gets_a_socket_of_its_own_and_no_image_of_a_batch_queues_behind_it(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.source_delay = 0.05
    async with worker(compute) as endpoint:
        live = await endpoint.load_from("rtsp://camera.example/stream")
        assert await live.predict() == prediction(0)
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(8)]
        results = await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 5)
        await live.cancel()
    assert all(r == [prediction(0), prediction(1)] for r in results)
    assert [s["url"] for s in pipeline.sockets[0].sources] == ["rtsp://camera.example/stream"]
    assert len(pipeline.sockets) == 1 + settings.pipeline_sockets


async def test_a_source_refused_for_another_reason_takes_no_room_on_the_socket(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, one_socket: None):
    pipeline.refusals = ["invalid_source"]
    async with worker(compute) as endpoint:
        with pytest.raises(PipelineCommandException):
            await predictions(await endpoint.load_from("https://images.example/a.jpg"))
        held = pipeline_backend(endpoint)._sockets[0].held
    assert held == set()


async def test_a_job_waiting_for_room_goes_on_whichever_socket_frees_first(
        compute: FakeComputeAPI, pipeline: FakePipelineAPI, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "pipeline_sockets", 2)
    pipeline.queue_full_at = 1
    async with worker(compute) as endpoint:
        pipeline.source_delay = 1.5
        slow = await endpoint.load_from("https://images.example/slow.jpg")
        await asyncio.sleep(0.05)
        pipeline.source_delay = 0.1
        quick = await endpoint.load_from("https://images.example/quick.jpg")
        waiting = await endpoint.load_from("https://images.example/waiting.jpg")
        result = await asyncio.wait_for(predictions(waiting), 1)
        await asyncio.gather(predictions(slow), predictions(quick))
    assert result == [prediction(0), prediction(1)]


async def test_sockets_the_pipeline_api_closes_together_are_reopened_once(compute: FakeComputeAPI,
                                                                         pipeline: FakePipelineAPI):
    pipeline.source_delay = 0.2
    async with worker(compute) as endpoint:
        jobs = [await endpoint.load_from(f"https://images.example/{n}.jpg") for n in range(8)]
        await asyncio.wait_for(asyncio.gather(*(predictions(job) for job in jobs)), 5)
        backend = pipeline_backend(endpoint)
        await asyncio.gather(*(run.close(1001, "shutting_down") for run in pipeline.runs))
        await asyncio.wait_for(_until(lambda: len(backend._open_sockets()) == 1), 5)
    assert backend._reconnects == 1


async def _until(condition: Callable[[], bool]):
    while not condition():
        await asyncio.sleep(0.01)


# 8. Credential


async def test_an_access_token_is_sent_as_given_on_the_upgrade(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    async with worker(compute, api_key=None, access_token="a user token"):
        pass
    assert pipeline.sockets[0].authorization == "Bearer a user token"
    assert compute.exchanges == []


async def test_the_api_key_is_exchanged_and_never_reaches_the_pipeline_api(compute: FakeComputeAPI,
                                                                         pipeline: FakePipelineAPI):
    async with worker(compute):
        pass
    assert compute.exchanges == ["Bearer the api key"]
    assert pipeline.sockets[0].authorization == "Bearer minted-1"


async def test_a_401_on_the_upgrade_exchanges_the_api_key_again(compute: FakeComputeAPI, pipeline: FakePipelineAPI):
    pipeline.upgrade_answers = [(401, {})]
    async with worker(compute):
        pass
    assert pipeline.sockets[0].authorization == "Bearer minted-2"


async def test_a_token_about_to_expire_is_exchanged_before_the_next_upgrade(pipeline: FakePipelineAPI):
    pipeline.idle_after_done = True
    async with FakeComputeAPI(pipeline.url, expires_in=30) as compute:
        async with worker(compute) as endpoint:
            await predictions(await endpoint.load_from("https://images.example/a.jpg"))
            await asyncio.sleep(0.05)
            await predictions(await endpoint.load_from("https://images.example/b.jpg"))
    assert [s.authorization for s in pipeline.sockets] == ["Bearer minted-2", "Bearer minted-3"]


# The sync wrapper


def test_the_sync_worker_runs_on_the_socket(tmp_path: Path):
    async def serve(ready: asyncio.Queue, stop: asyncio.Event):
        async with FakePipelineAPI() as pipeline, FakeComputeAPI(pipeline.url) as compute:
            await ready.put(compute.url)
            await stop.wait()

    loop = asyncio.new_event_loop()
    ready: asyncio.Queue = asyncio.Queue()
    stop = asyncio.Event()
    thread = threading.Thread(target=loop.run_until_complete, args=(serve(ready, stop),), daemon=True)
    thread.start()
    url = asyncio.run_coroutine_threadsafe(ready.get(), loop).result(5)
    try:
        with EyePopSdk.sync_worker(eyepop_url=url, api_key="k", is_pipeline_mode=True, pop=PERSON) as endpoint:
            job = endpoint.load_from("https://images.example/a.jpg")
            assert job.predict() == prediction(0)
            assert job.predict() == prediction(1)
            assert job.predict() is None
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(5)
