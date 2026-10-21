import asyncio
import contextlib
import pathlib
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web

import eyepop.pipeline
from eyepop.pipeline import (
    STOP,
    Ability,
    End,
    EyePop,
    EyePopError,
    Prediction,
    Result,
    Started,
)
from eyepop.pipeline import session as session_module
from tests.pipeline.fakes import (
    PREDICTIONS,
    FakePipelineAPI,
    Received,
    closes,
    receive,
    serve,
    until_closed,
)

JWT = "header.payload.signature"
PERSON = Ability("eyepop.person")
POSE = Ability("eyepop.person").crop(Ability("eyepop.person.2d-body-points"))


@pytest.fixture
async def api() -> AsyncIterator[FakePipelineAPI]:
    async with FakePipelineAPI() as fake:
        yield fake
    for upgrade in fake.upgrades:
        assert "eyepop.v1" in upgrade.protocols
        assert upgrade.authorization == f"Bearer {JWT}"


@pytest.fixture
async def ep(api: FakePipelineAPI) -> AsyncIterator[EyePop]:
    async with EyePop(api.url, access_token=JWT, wait=2) as client:
        yield client


async def wait_for(condition, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


# Opening


async def test_the_pop_is_set_before_the_pipeline_opens_to_the_caller(api: FakePipelineAPI, ep: EyePop):
    """`async with` answers once the first `setPop` has its result; a pipeline with no pop keeps the default."""
    async with ep.pipeline(POSE):
        assert api.pops == [POSE.json]
    async with ep.pipeline():
        assert api.pops == [POSE.json]


async def test_a_pop_the_caller_cannot_load_fails_the_open(api: FakePipelineAPI, ep: EyePop):
    """The first pop's error is raised by `async with`, with the server's code and message."""
    api.pop_error = "ability_unavailable"
    with pytest.raises(EyePopError) as raised:
        async with ep.pipeline(PERSON):
            pytest.fail("opened")
    assert (raised.value.error, raised.value.message) == ("ability_unavailable", "no such ability")


async def test_a_pipeline_opens_once(ep: EyePop):
    """A pipeline that was closed cannot be opened again."""
    p = ep.pipeline(PERSON)
    async with p:
        pass
    with pytest.raises(RuntimeError):
        async with p:
            pass


# Sources


async def test_each_source_yields_its_own_predictions_typed_with_their_source(api: FakePipelineAPI, ep: EyePop):
    """Two sources on one pipeline: each iterator gets its predictions only, each prediction knows its source."""
    async with ep.pipeline(PERSON) as p:
        a = p.source("https://example.com/a.jpg", id="a")
        b = p.source("https://example.com/b.jpg", id="b")
        frames_b = [f async for f in b]
        frames_a = [f async for f in a]
    assert [(f.source, f.raw["n"]) for f in frames_a] == [("a", 0), ("a", 1)]
    assert [f.source for f in frames_b] == ["b", "b"]
    assert frames_a[0].objects[0].label == "person"
    setSources = [m for m in api.received[0].messages if isinstance(m, dict) and m["type"] == "setSource"]
    assert [(m["id"], m["url"]) for m in setSources] == [("a", "https://example.com/a.jpg"), ("b", "https://example.com/b.jpg")]


async def test_an_id_open_twice_is_refused_and_a_closed_pipeline_takes_no_source(ep: EyePop):
    """A source id is unique among the open ones; nothing is set on a closed pipeline."""
    async with ep.pipeline(PERSON) as p:
        p.source("https://example.com/a.jpg", id="a")
        with pytest.raises(ValueError):
            p.source("https://example.com/b.jpg", id="a")
    with pytest.raises(EyePopError):
        p.source("https://example.com/c.jpg")


async def test_a_source_that_fails_raises_its_message_and_the_pipeline_goes_on(api: FakePipelineAPI, ep: EyePop):
    """A source that ends with `error` raises for it alone; the next one plays on the same socket."""
    api.failing.add("bad")
    async with ep.pipeline(PERSON) as p:
        with pytest.raises(EyePopError) as raised:
            [f async for f in p.source("https://example.com/x.jpg", id="bad")]
        assert raised.value.message == "bad media"
        assert len([f async for f in p.source("https://example.com/y.jpg", id="good")]) == 2


async def test_batch_yields_each_prediction_and_each_failure_with_its_source(api: FakePipelineAPI, ep: EyePop):
    """A url, an asset and a url queue on one socket; each prediction and each failed image comes back with its source."""
    api.frames = 1
    images = ["https://example.com/a.jpg", {"assetUuid": "4d1e"}, "https://example.com/c.jpg"]
    async with ep.pipeline(PERSON) as p:
        api.failing.add("s2")  # the second image's id
        results = [r async for r in p.batch(images)]
    assert [r.source for r in results] == images
    assert results[0].prediction is not None and results[0].prediction.source == "s1"
    assert results[1] == Result(images[1], error="bad media")
    sent = [m for m in api.received[0].messages if isinstance(m, dict) and m["type"] == "setSource"]
    assert [m.get("assetUuid") for m in sent] == [None, "4d1e", None]


async def test_live_and_preempt_are_sent_as_members(api: FakePipelineAPI, ep: EyePop):
    """`stream` is live; `mode="preempt"` ends what runs, and the preempted source's iterator ends without raising."""
    async with ep.pipeline(PERSON) as p:
        cam = p.stream("rtsp://camera/1", id="cam")
        frames = aiter(cam)
        await anext(frames)
        clip = p.video("https://example.com/clip.mp4", id="clip", mode="preempt", fps=5)
        assert len([f async for f in clip]) == 2
        async for frame in frames:  # the live source was preempted: what was in flight arrives, then it ends
            assert frame.source == "cam"
    cams = [m for m in api.received[0].messages if isinstance(m, dict) and m.get("id") in ("cam", "clip")]
    assert cams[0]["live"] is True and "mode" not in cams[0]
    assert cams[1]["mode"] == "preempt" and cams[1]["fps"] == 5


async def test_dropped_predictions_are_counted_on_the_source(api: FakePipelineAPI):
    """`dropped` messages add up on the live source's `dropped`."""
    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        received = Received()
        while (m := await receive(ws, received)) is not None:
            if m["type"] == "setPop":
                await ws.send_json({"type": "result", "id": m["id"]})
            elif m["type"] == "setSource":
                await ws.send_json({"type": "result", "id": m["id"]})
                await ws.send_json({"type": "dropped", "id": m["id"], "count": 3})
                await ws.send_json({"type": "dropped", "id": m["id"], "count": 4})
                await ws.send_json({"type": "end", "id": m["id"], "reason": "eos"})

    api.script = script
    async with EyePop(api.url, access_token=JWT) as client, client.pipeline(PERSON) as p:
        cam = p.stream("rtsp://camera/1")
        assert [f async for f in cam] == []
        assert cam.dropped == 7


# Uploads


async def test_media_that_fits_goes_in_one_message_typed_from_its_name(api: FakePipelineAPI, ep: EyePop, tmp_path: pathlib.Path):
    """A path, bytes and a file: one combined message each, with no `send` or `uploaded`."""
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"mp4 bytes")
    with (tmp_path / "pic.jpg").open("wb+") as f:
        f.write(b"jpeg bytes")
        f.seek(0)
        async with ep.pipeline(PERSON) as p:
            frames = [f async for f in p.video(path, id="clip")]
            [x async for x in p.source(b"png bytes", id="raw", content_type="image/png")]
            [x async for x in p.source(f, id="pic")]
    assert len(frames) == 2
    assert api.media == {"clip": b"mp4 bytes", "raw": b"png bytes", "pic": b"jpeg bytes"}
    sent = [m for m in api.received[0].messages if isinstance(m, bytes)]
    assert all(m[:1] == b"\x02" for m in sent)
    assert [m["type"] for m in api.received[0].messages if isinstance(m, dict)].count("uploaded") == 0


async def test_media_over_the_limit_goes_chunked_and_within_the_credit(api: FakePipelineAPI, ep: EyePop, monkeypatch: pytest.MonkeyPatch):
    """Over the combined limit the bytes follow `send`, in pieces no larger than the credit the server holds."""
    monkeypatch.setattr(session_module, "_COMBINED_MAX", 100)
    api.window = 300
    media = bytes(range(256)) * 4
    async with ep.pipeline(PERSON) as p:
        frames = [f async for f in p.video(media, id="v", content_type="video/mp4")]
    assert len(frames) == 2
    messages = api.received[0].messages
    chunks = [m for m in messages if isinstance(m, bytes)]
    assert all(m[:1] == b"\x01" and len(m) - 1 <= 300 for m in chunks) and len(chunks) >= 4
    assert b"".join(m[1:] for m in chunks) == media == api.media["v"]
    assert messages[-1] == {"type": "uploaded", "id": messages[-1]["id"]}


async def test_a_chunked_upload_waits_for_credit(api: FakePipelineAPI, ep: EyePop, monkeypatch: pytest.MonkeyPatch):
    """The client sends exactly the credit it holds and then nothing until the server grants more."""
    monkeypatch.setattr(session_module, "_COMBINED_MAX", 0)
    received = Received()

    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        while (m := await receive(ws, received)) is not None:
            if m["type"] == "setPop":
                await ws.send_json({"type": "result", "id": m["id"]})
            elif m["type"] == "setSource":
                await ws.send_json({"type": "result", "id": m["id"]})
                await ws.send_json({"type": "send", "id": m["id"], "credit": 10})
                assert len(await receive(ws, received)) == 11
                assert await receive(ws, received, timeout=0.3) is None  # no credit left: it waits
                await ws.send_json({"type": "credit", "id": m["id"], "bytes": 5})
                assert len(await receive(ws, received)) == 6
                assert await receive(ws, received, timeout=0.3) is None
                await ws.send_json({"type": "credit", "id": m["id"], "bytes": 1000})
                assert await receive(ws, received) is not None
                assert (await receive(ws, received)) == {"type": "uploaded", "id": m["id"]}
                await ws.send_json({"type": "end", "id": m["id"], "reason": "eos"})

    api.script = script
    async with ep.pipeline(PERSON) as p:
        assert [f async for f in p.video(b"x" * 100, id="v")] == []
    sizes = [len(m) - 1 for m in received.messages if isinstance(m, bytes)]
    assert sizes == [10, 5, 85]


async def test_an_upload_that_ends_first_stops_its_bytes_and_sends_uploaded(api: FakePipelineAPI, ep: EyePop, monkeypatch: pytest.MonkeyPatch):
    """The source ends while the media is still being read: no more bytes follow, and `uploaded` lets the next upload start."""
    monkeypatch.setattr(session_module, "_COMBINED_MAX", 0)
    more = asyncio.Event()
    read_after_end: list[bool] = []

    async def stalled() -> AsyncIterator[bytes]:
        yield b"first chunk"
        await more.wait()  # never set: the source's end must stop the sending
        read_after_end.append(True)
        yield b"never sent"

    received = Received()

    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        while (m := await receive(ws, received)) is not None:
            if m["type"] == "setPop":
                await ws.send_json({"type": "result", "id": m["id"]})
            elif m["type"] == "setSource":
                await ws.send_json({"type": "send", "id": m["id"], "credit": 1000})
                assert await receive(ws, received) == b"\x01first chunk"
                await ws.send_json({"type": "end", "id": m["id"], "reason": "error", "message": "stalled"})
                assert await receive(ws, received, timeout=2) == {"type": "uploaded", "id": m["id"]}
                assert await receive(ws, received, timeout=0.3) is None

    api.script = script
    async with ep.pipeline(PERSON) as p:
        with pytest.raises(EyePopError, match="stalled"):
            [f async for f in p.video(stalled(), id="d")]
        await wait_for(lambda: len(received.messages) >= 4)
    assert read_after_end == []


# Commands


async def test_set_pop_is_answered_and_the_source_keeps_running(api: FakePipelineAPI, ep: EyePop):
    """`set_pop` sends the new pop and returns on its result; the live source goes on."""
    async with ep.pipeline(PERSON) as p:
        frames = aiter(p.stream("rtsp://camera/1"))
        await anext(frames)
        await p.set_pop(POSE)
        await anext(frames)
        await frames.aclose()
    assert api.pops == [PERSON.json, POSE.json]


async def test_a_pop_that_fails_raises_for_that_call_and_the_socket_stays_open(api: FakePipelineAPI, ep: EyePop):
    """A failed `setPop` raises its error, and the pipeline still takes sources."""
    async with ep.pipeline(PERSON) as p:
        api.pop_error = "invalid_pop"
        with pytest.raises(EyePopError) as raised:
            await p.set_pop({"components": []})
        assert raised.value.error == "invalid_pop"
        assert len([f async for f in p.source("https://example.com/a.jpg")]) == 2


async def test_graph_is_text_for_dot_and_bytes_for_the_rest(ep: EyePop):
    """`getGraph` as `dot` is a `str`; `png` is base64 decoded to `bytes`."""
    async with ep.pipeline(PERSON) as p:
        assert await p.graph("dot") == "digraph {}"
        assert await p.graph("png", verbose=True) == b"\x89PNG"


async def test_a_command_sent_before_the_pipeline_opens_waits_for_it(api: FakePipelineAPI, ep: EyePop):
    """A source set before `async with` is sent after the first pop, in order."""
    p = ep.pipeline(PERSON)
    cam = p.source("https://example.com/a.jpg", id="a")
    async with p:
        assert len([f async for f in cam]) == 2
    types = [m["type"] for m in api.received[0].messages if isinstance(m, dict)]
    assert types[:2] == ["setPop", "setSource"]


# Events


async def test_events_is_every_event_in_order_typed(api: FakePipelineAPI, ep: EyePop):
    """The tap yields `Started`, `Prediction`, `End` and `Done`, and raises nothing on a clean end."""
    async with ep.pipeline(PERSON) as p:
        events = p.events()
        p.source("https://example.com/a.jpg", id="a")
        seen = [e async for e in _until_done(events)]
    assert [type(e) for e in seen] == [Started, Prediction, Prediction, End]
    assert seen[1].source == "a" and seen[3] == End("a", "eos")  # type: ignore[union-attr]


async def _until_done(events: AsyncIterator[Any]) -> AsyncIterator[Any]:
    async for event in events:
        if isinstance(event, eyepop.pipeline.Done):
            return
        yield event


# Handlers


async def test_run_calls_the_handlers_and_waits_for_every_source(api: FakePipelineAPI, ep: EyePop):
    """Two sources queued back to back both play, though `Done` can come between them; handlers of a source see only it."""
    pipe = ep.pipeline(PERSON)
    seen: list[str] = []
    only_b: list[str] = []

    @pipe.on_prediction
    def each(frame: Prediction) -> None:
        seen.append(f"{frame.source}:{frame.raw['n']}")

    @pipe.on_prediction(source="b")
    async def b_only(frame: Prediction) -> None:
        only_b.append(frame.source or "")

    @pipe.on_end
    def ended(end: End) -> None:
        seen.append(f"end {end.source}")

    pipe.source("https://example.com/a.jpg", id="a")
    pipe.source("https://example.com/b.jpg", id="b")
    await pipe.arun()
    assert seen == ["a:0", "a:1", "end a", "b:0", "b:1", "end b"]
    assert only_b == ["b", "b"]


async def test_a_handler_may_set_a_source_and_run_waits_for_it(ep: EyePop):
    """A source added from a handler is waited for as well."""
    pipe = ep.pipeline(PERSON)
    seen: list[str] = []

    @pipe.on_end
    def ended(end: End) -> None:
        seen.append(end.source)
        if end.source == "a":
            pipe.source("https://example.com/b.jpg", id="b")

    pipe.source("https://example.com/a.jpg", id="a")
    await pipe.arun()
    assert seen == ["a", "b"]


async def test_stop_from_a_handler_ends_the_run_and_closes_the_socket(api: FakePipelineAPI, ep: EyePop):
    """`STOP` ends `arun` while the live source still runs, and the socket closes with 1000."""
    pipe = ep.pipeline(PERSON)
    seen: list[int] = []

    @pipe.on_prediction
    def each(frame: Prediction) -> Any:
        seen.append(frame.raw["n"])
        if len(seen) == 3:
            return STOP

    pipe.stream("rtsp://camera/1")
    await pipe.arun()
    assert seen == [0, 1, 2]
    await wait_for(lambda: api.received[0].close_code is not None)
    assert api.received[0].close_code == 1000


async def test_run_without_a_source_is_refused(ep: EyePop):
    """Nothing to run is a mistake, not an idle socket."""
    with pytest.raises(ValueError):
        await ep.pipeline(PERSON).arun()


# Closing


async def test_breaking_a_stream_closes_its_socket(api: FakePipelineAPI, ep: EyePop):
    """Breaking out of a one-shot stream closes the pipeline's socket with 1000."""
    seen = []
    async with contextlib.aclosing(ep.use(PERSON).stream("rtsp://camera/1")) as stream:
        async for frame in stream:
            seen.append(frame.raw["n"])
            if len(seen) == 4:
                break
    await wait_for(lambda: bool(api.received) and api.received[0].close_code is not None)
    assert api.received[0].close_code == 1000
    assert seen == [0, 1, 2, 3]


async def test_closing_the_client_ends_a_stream(api: FakePipelineAPI):
    """Closing the client closes its pipelines with 1000 and ends the iteration without an error."""
    client = EyePop(api.url, access_token=JWT)
    seen = []
    async for frame in client.use(PERSON).stream("rtsp://camera/1"):
        seen.append(frame)
        if len(seen) == 3:
            await client.close()
    assert len(seen) == 3
    await wait_for(lambda: api.received[0].close_code is not None)
    assert api.received[0].close_code == 1000


# Capacity and failure


def refused_for_capacity(times: int, error: str = "overloaded"):
    """A script that refuses the live source of the first `times` sockets as a full replica does."""

    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        if len(fake.upgrades) > times:
            return await serve(ws, fake)
        received = Received()
        while (m := await receive(ws, received)) is not None:
            if m["type"] == "setPop":
                await ws.send_json({"type": "result", "id": m["id"]})
            elif m["type"] == "setSource":
                await ws.send_json({"type": "error", "id": m["id"], "error": error, "message": "no capacity"})
                await ws.close(code=1011, message=error.encode())

    return script


async def test_an_upgrade_answered_503_is_retried(api: FakePipelineAPI, ep: EyePop):
    """An upgrade answered 503 with Retry-After is retried and the pipeline opens."""
    api.upgrade_answers.append((503, {"error": "pipeline_start_failed", "retry_after_seconds": 1}, {"Retry-After": "1"}))
    async with ep.pipeline(PERSON):
        pass
    assert len(api.upgrades) == 2


async def test_a_refused_upgrade_raises_at_once(api: FakePipelineAPI, ep: EyePop):
    """An upgrade answered 403 raises with its status and no error."""
    api.upgrade_answers.append((403, "Forbidden", {}))
    with pytest.raises(EyePopError) as raised:
        async with ep.pipeline(PERSON):
            pass
    assert (raised.value.status, raised.value.error, raised.value.close_code) == (403, None, None)
    assert len(api.upgrades) == 1


@pytest.mark.parametrize("error", ["overloaded", "pipeline_start_failed"])
async def test_a_one_shot_call_refused_for_capacity_reopens_and_runs(api: FakePipelineAPI, ep: EyePop, error: str):
    """A live source refused for want of capacity, before anything was yielded, is run again on a new socket."""
    api.script = refused_for_capacity(1, error)
    async for frame in ep.use(PERSON).stream("rtsp://camera/1"):
        assert frame.source is not None
        break
    assert len(api.upgrades) == 2


async def test_a_one_shot_call_raises_once_wait_has_passed(api: FakePipelineAPI):
    """Still refused after `wait`, the last refusal is raised."""
    api.script = refused_for_capacity(10)
    async with EyePop(api.url, access_token=JWT, wait=1.5) as client:
        with pytest.raises(EyePopError) as raised:
            [f async for f in client.use(PERSON).stream("rtsp://camera/1")]
    assert raised.value.error == "overloaded"
    assert len(api.upgrades) == 2


async def test_a_kept_pipeline_never_reopens_itself(api: FakePipelineAPI, ep: EyePop):
    """A pipeline the caller keeps raises the refusal: it does not know what the caller has seen."""
    api.script = refused_for_capacity(10)
    async with ep.pipeline(PERSON) as p:
        with pytest.raises(EyePopError) as raised:
            [f async for f in p.stream("rtsp://camera/1")]
    assert raised.value.error == "overloaded"
    assert len(api.upgrades) == 1


@pytest.mark.parametrize(
    ("script", "close_code", "error"),
    [
        (closes(1011), 1011, None),
        (closes(1001, b"shutting_down"), 1001, "shutting_down"),
        (closes(1008, b"credential_expired"), 1008, "credential_expired"),
    ],
)
async def test_a_close_other_than_1000_raises_with_its_code_and_reason(api: FakePipelineAPI, ep: EyePop, script, close_code: int, error: str | None):
    """A socket the server closes for a reason raises with the close code and the reason."""
    api.script = script
    with pytest.raises(EyePopError) as raised:
        async with ep.pipeline(PERSON):
            pass
    assert (raised.value.status, raised.value.close_code, raised.value.error) == (None, close_code, error)


async def test_the_pipeline_stopping_ends_every_open_source_with_the_failure(api: FakePipelineAPI, ep: EyePop):
    """A socket closed 1011 `pipeline_stopped` while sources are open raises in each iterator and in the tap."""
    received = Received()

    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        while (m := await receive(ws, received)) is not None:
            if m["type"] == "setPop":
                await ws.send_json({"type": "result", "id": m["id"]})
            elif m["type"] == "setSource" and m["id"] == "b":
                await ws.close(code=1011, message=b"pipeline_stopped")

    api.script = script
    async with ep.pipeline(PERSON) as p:
        events = p.events()
        a, b = p.stream("rtsp://camera/1", id="a"), p.stream("rtsp://camera/2", id="b")
        for source in (a, b):
            with pytest.raises(EyePopError) as raised:
                [f async for f in source]
            assert raised.value.error == "pipeline_stopped"
        with pytest.raises(EyePopError):
            [e async for e in events]


@pytest.mark.parametrize("text", ["not json", "[1, 2]"])
async def test_a_message_that_is_not_a_json_object_raises_and_closes_the_socket(api: FakePipelineAPI, ep: EyePop, text: str):
    """A text frame that is not a JSON object fails the pipeline and closes the socket with 1000."""
    received = Received()

    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        await ws.send_str(text)
        await until_closed(ws, received)

    api.script = script
    with pytest.raises(EyePopError):
        async with ep.pipeline(PERSON):
            pass
    assert received.close_code == 1000


async def test_a_prediction_that_is_not_an_object_fails_the_pipeline(api: FakePipelineAPI, ep: EyePop):
    """A prediction message whose prediction is not a JSON object raises EyePopError."""
    async def script(ws: web.WebSocketResponse, fake: FakePipelineAPI) -> None:
        received = Received()
        while (m := await receive(ws, received)) is not None:
            await ws.send_json({"type": "result", "id": m["id"]})
            if m["type"] == "setSource":
                await ws.send_json({"type": "prediction", "id": m["id"], "prediction": "not one"})

    api.script = script
    async with ep.pipeline(PERSON) as p:
        with pytest.raises(EyePopError):
            [f async for f in p.source("https://example.com/a.jpg")]


# The module's functions and decorators


@pytest.fixture
def configured(api: FakePipelineAPI, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> FakePipelineAPI:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EYEPOP_API_KEY", raising=False)
    monkeypatch.setenv("EYEPOP_PIPELINE_URL", api.url)
    monkeypatch.setenv("EYEPOP_ACCESS_TOKEN", JWT)
    return api


async def test_the_functions_need_no_client(configured: FakePipelineAPI):
    """`image`, `batch`, `stream` and `video` take their url and token from the environment and close what they open."""
    assert await eyepop.pipeline.image(PERSON, {"url": "https://example.com/a.jpg"}) == Prediction.parse(PREDICTIONS[0])
    results = [r async for r in eyepop.pipeline.batch(PERSON, ["https://example.com/a.jpg"])]
    assert [r.source for r in results] == ["https://example.com/a.jpg"] * 2
    assert len([f async for f in eyepop.pipeline.video(PERSON, b"clip")]) == 2
    async for frame in eyepop.pipeline.stream(PERSON, "rtsp://camera/1"):
        assert frame.source is not None
        break
    async with eyepop.pipeline.pipeline(PERSON) as p:
        assert await p.graph() == "digraph {}"


async def test_a_decorated_function_is_the_job(configured: FakePipelineAPI):
    """`@stream(...)` turns the function into a job: it gets each frame, and returning `STOP` ends it."""
    frames: list[int] = []

    @eyepop.pipeline.stream(PERSON, "rtsp://camera/1")
    async def watch(frame: Prediction) -> Any:
        frames.append(frame.raw["n"])
        if len(frames) == 4:
            return STOP

    await watch()
    assert frames == [0, 1, 2, 3]


async def test_decorated_jobs_run_together_with_sync_or_async_handlers(configured: FakePipelineAPI):
    """`run` runs jobs side by side; a handler may be a plain function."""
    seen: list[Any] = []

    @eyepop.pipeline.batch(PERSON, ["https://example.com/a.jpg"])
    def each(result: Result) -> None:
        seen.append(result.source)

    @eyepop.pipeline.video(PERSON, b"clip")
    async def clip(frame: Prediction) -> None:
        seen.append(frame.source)

    await eyepop.pipeline.run(each, clip)
    assert sorted(seen) == sorted(["https://example.com/a.jpg"] * 2 + ["s1"] * 2)
