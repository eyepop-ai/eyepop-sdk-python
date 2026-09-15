"""The relay example's lifecycle: reconnect, shutdown, and error attribution.

Drives `relay_rtsp_source` against fakes rather than a camera. The acceptance
criteria on AWSU-255 are all about what happens when things go wrong - the
camera drops, the caller walks away, the upload fails - and none of those are
reachable from a happy-path run against a real stream.
"""

from __future__ import annotations

import asyncio
from fractions import Fraction

import av
import numpy as np
import pytest

from examples.relay_example import (
    CameraError,
    MuxError,
    UploadError,
    _relay_one_session,
    relay_rtsp_source,
)


class FakeJob:
    """Yields the given results, then None for end of stream."""

    def __init__(self, results: list[dict] | None = None, error: Exception | None = None):
        self._results = list(results or [])
        self._error = error

    async def predict(self):
        if self._results:
            return self._results.pop(0)
        if self._error is not None:
            raise self._error
        return None


class FakeEndpoint:
    def __init__(self, jobs: list[FakeJob] | None = None, upload_error: Exception | None = None):
        self._jobs = list(jobs or [])
        self._upload_error = upload_error
        self.upload_count = 0

    async def upload_stream(self, *args, **kwargs):
        self.upload_count += 1
        if self._upload_error is not None:
            raise self._upload_error
        return self._jobs.pop(0) if self._jobs else FakeJob()


@pytest.fixture
def session_recorder(monkeypatch):
    """Replaces one RTSP session with a scripted outcome.

    `_relay_one_session` is where av and the muxing live; everything this module
    tests sits above it, in the loop that decides whether to try again.
    """
    calls: list[int] = []

    def install(outcomes):
        async def fake_session(source_url, endpoint, **kwargs):
            attempt = len(calls)
            calls.append(attempt)
            outcome = outcomes[min(attempt, len(outcomes) - 1)]
            for result in outcome.get("results", []):
                yield result
            error = outcome.get("error")
            if error is not None:
                raise error

        monkeypatch.setattr("examples.relay_example._relay_one_session", fake_session)
        return calls

    return install


async def drain(generator, limit: int = 100) -> list[dict]:
    results = []
    async for item in generator:
        results.append(item)
        if len(results) >= limit:
            break
    return results


@pytest.mark.asyncio
async def test_camera_drop_reconnects_and_keeps_yielding(session_recorder):
    calls = session_recorder([
        {"results": [{"seq": 1}], "error": CameraError("camera went away")},
        {"results": [{"seq": 2}]},
    ])

    results = await drain(relay_rtsp_source("rtsp://camera.invalid/s", FakeEndpoint()))

    # Predictions from both sessions reach the caller as one stream: a
    # reconnect is not visible as an end of iteration.
    assert results == [{"seq": 1}, {"seq": 2}]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_camera_drop_raises_when_reconnect_is_off(session_recorder):
    session_recorder([{"results": [{"seq": 1}], "error": CameraError("camera went away")}])

    generator = relay_rtsp_source("rtsp://camera.invalid/s", FakeEndpoint(), reconnect=False)
    with pytest.raises(CameraError):
        await drain(generator)


@pytest.mark.asyncio
async def test_a_clean_end_of_source_does_not_reconnect(session_recorder):
    # A finite stream that ends is done. Reconnecting would replay it forever,
    # which is the failure mode a naive retry loop has.
    calls = session_recorder([{"results": [{"seq": 1}]}])

    results = await drain(relay_rtsp_source("rtsp://camera.invalid/s", FakeEndpoint()))

    assert results == [{"seq": 1}]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_an_upload_failure_is_not_retried_as_a_camera_drop(session_recorder):
    # The camera is fine; retrying hammers a worker that already refused.
    session_recorder([{"error": UploadError("worker refused the stream")}])

    with pytest.raises(UploadError):
        await drain(relay_rtsp_source("rtsp://camera.invalid/s", FakeEndpoint()))


class StopRetrying(Exception):
    """Breaks an intentionally endless reconnect loop at a known point."""


@pytest.mark.asyncio
async def test_backoff_grows_and_is_capped_while_nothing_is_delivered(
    session_recorder, monkeypatch
):
    """A camera that connects and immediately fails must not spin.

    Resetting the backoff per attempt rather than per delivered result turns
    this into a tight retry loop against a camera that is up but broken.
    """
    slept: list[float] = []

    # The loop under test is deliberately endless - a camera that never comes
    # back is retried forever by design - and patching sleep removes the only
    # thing pacing it. Ending it from here rather than by counting yields: the
    # generator yields nothing in this scenario, so anything waiting on a
    # result waits forever while `slept` grows without bound.
    async def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) >= 5:
            raise StopRetrying

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    session_recorder([{"error": CameraError("accepts then fails")}])

    with pytest.raises(StopRetrying):
        await drain(relay_rtsp_source(
            "rtsp://camera.invalid/s", FakeEndpoint(), max_backoff_s=8.0,
        ))

    # Doubling, then held at the cap rather than growing without limit.
    assert slept == [1.0, 2.0, 4.0, 8.0, 8.0], slept


@pytest.mark.asyncio
async def test_backoff_resets_once_a_session_delivers(session_recorder, monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    session_recorder([
        {"error": CameraError("first attempt fails")},
        {"error": CameraError("second fails too")},
        {"results": [{"seq": 1}], "error": CameraError("drops after delivering")},
        {"results": [{"seq": 2}]},
    ])

    results = await drain(relay_rtsp_source("rtsp://camera.invalid/s", FakeEndpoint()))

    assert results == [{"seq": 1}, {"seq": 2}]
    # 1s, 2s while nothing was delivered; back to 1s after a session produced a
    # prediction.
    assert slept == [1.0, 2.0, 1.0], slept


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
async def test_an_unusable_backoff_ceiling_is_rejected(invalid):
    """An unusable ceiling is rejected at the call.

    A negative ceiling makes every sleep negative, and asyncio.sleep returns
    immediately for those - an unthrottled retry loop against a camera that is
    down, discovered as a busy process rather than as an error.
    """
    with pytest.raises(ValueError):
        await drain(relay_rtsp_source("rtsp://camera.invalid/s", FakeEndpoint(), max_backoff_s=invalid))


@pytest.mark.asyncio
async def test_a_ceiling_below_the_opening_delay_still_applies(session_recorder, monkeypatch):
    # The opening delay is a constant, so a caller asking for a ceiling under
    # it would otherwise wait longer than they allowed, on the first retry and
    # again after every success.
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        if len(slept) >= 3:
            raise StopRetrying

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    session_recorder([{"error": CameraError("down")}])

    with pytest.raises(StopRetrying):
        await drain(relay_rtsp_source(
            "rtsp://camera.invalid/s", FakeEndpoint(), max_backoff_s=0.25,
        ))

    assert slept == [0.25, 0.25, 0.25], slept


@pytest.fixture
def h264_file(tmp_path):
    """A short H.264 file, so the muxing path runs for real."""
    path = tmp_path / "source.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=25)
    stream.width, stream.height = 160, 120
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "g": "25"}
    for index in range(25):
        image = np.full((120, 160, 3), index * 8 % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
        frame.pts = index
        frame.time_base = Fraction(1, 25)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.mark.asyncio
async def test_a_failed_mux_close_is_reported_rather_than_read_as_a_clean_end(
    h264_file, monkeypatch
):
    """close() flushes what is still buffered, so a failure truncates the upload.

    Everything here is real except close(): the file is demuxed, remuxed to
    MPEG-TS and read back. Only the final flush is made to fail, which is the
    one path that used to end the stream silently.
    """
    real_open = av.open

    def open_with_failing_close(*args, **kwargs):
        output = real_open(*args, **kwargs)
        if kwargs.get("mode") == "w":
            class FlushFails:
                def __getattr__(self, name):
                    return getattr(output, name)

                def close(self):
                    raise RuntimeError("no space left on device")

            return FlushFails()
        return output

    monkeypatch.setattr(av, "open", open_with_failing_close)

    class DrainEndpoint:
        async def upload_stream(self, pipe, **kwargs):
            class Job:
                async def predict(self):
                    return None if not await asyncio.to_thread(pipe.read, 65536) else {}
            return Job()

    with pytest.raises(MuxError):
        await drain(_relay_one_session(str(h264_file), DrainEndpoint()))


@pytest.mark.asyncio
async def test_a_live_source_ending_without_an_error_is_a_drop(h264_file):
    """A camera that stops sending raises rather than ending quietly.

    Measured against a real camera (AWSU-258): severing an RTSP-over-TCP
    connection ends PyAV's demux generator normally and raises nothing. A file
    running out reaches `_relay_one_session` the same way, which is what makes
    it usable here - so this drives the exact code path a dropped camera takes.

    Without this, the drop is read as a finite stream finishing and
    `relay_rtsp_source` returns instead of reconnecting.
    """
    class DrainEndpoint:
        async def upload_stream(self, pipe, **kwargs):
            class Job:
                async def predict(self):
                    return None if not await asyncio.to_thread(pipe.read, 65536) else {}
            return Job()

    with pytest.raises(CameraError):
        await drain(_relay_one_session(str(h264_file), DrainEndpoint()))


@pytest.mark.asyncio
async def test_a_source_that_ends_is_retried_rather_than_ending_the_relay(h264_file, monkeypatch):
    """The end-to-end consequence: the relay tries again instead of stopping.

    The test above pins the raise; this pins that `reconnect=True` acts on it.
    Driven through the real session so the two cannot drift apart - a
    CameraError raised but not acted on would pass the first test alone.
    """
    attempts: list[int] = []

    class DrainEndpoint:
        async def upload_stream(self, pipe, **kwargs):
            attempts.append(len(attempts))
            class Job:
                async def predict(self):
                    return None if not await asyncio.to_thread(pipe.read, 65536) else {}
            return Job()

    async def fake_sleep(seconds):
        if len(attempts) >= 3:
            raise StopRetrying

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    with pytest.raises(StopRetrying):
        await drain(relay_rtsp_source(str(h264_file), DrainEndpoint()))

    assert len(attempts) >= 3, attempts


@pytest.mark.asyncio
async def test_a_caller_walking_away_is_not_reported_as_a_drop(h264_file):
    """Abandoning the generator stays a clean shutdown, not a reported drop.

    Treating every quiet end as a drop would turn ordinary teardown into an
    error. The usual path is already safe without the stop flag: a thread that
    sees `stop` breaks out of the demux loop, so the end-of-stream branch never
    runs, and a closed generator does not execute the code after its `finally`
    anyway.

    The `not stop.is_set()` guard covers only the narrow race where the source
    ends *after* the final stop check of the last iteration. Mutation testing
    confirms no test here fails without that guard, and reproducing the race
    deterministically would mean adding a seam to an example. It is kept as
    cheap correctness, not as something this test pins.
    """
    class SlowEndpoint:
        async def upload_stream(self, pipe, **kwargs):
            class Job:
                async def predict(self):
                    await asyncio.to_thread(pipe.read, 1024)
                    return {"seq": 1}
            return Job()

    generator = _relay_one_session(str(h264_file), SlowEndpoint())
    assert await drain(generator, limit=1) == [{"seq": 1}]
    # No CameraError escapes the close; a stop the caller asked for is not a
    # failure to report.
    await generator.aclose()
