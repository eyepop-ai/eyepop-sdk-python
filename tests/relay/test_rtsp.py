"""`eyepop.relay.rtsp` on its own terms - no endpoint, no worker, no upload.

Being able to write this file is the point of the refactor: the relay used to
be reachable only through something shaped like a `WorkerEndpoint`, so every
test of the muxing had to carry a fake one. Here the stream is just bytes.
"""

from __future__ import annotations

import av
import pytest
from av.error import FFmpegError

from eyepop.relay import (
    CameraError,
    MuxError,
    RtspRelayStream,
    rtsp_relay_stream,
)


async def collect(stream) -> bytes:
    return b"".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_it_yields_mpegts_bytes(h264_file):
    """The bytes are a real MPEG-TS, not an opaque blob."""
    stream = await rtsp_relay_stream(str(h264_file))
    data = await collect(stream)

    assert len(data) > 0
    # Every 188-byte TS packet starts with the sync byte, and the first one
    # starts at offset zero. Checking a few rather than one: a single 0x47 can
    # occur by chance in payload, a run of them at the right stride cannot.
    assert data[0] == 0x47
    assert all(data[offset] == 0x47 for offset in range(0, 188 * 20, 188))


@pytest.mark.asyncio
async def test_a_source_that_ends_is_reported_as_a_drop(h264_file):
    """A live source that stops sending has dropped - and says so afterwards.

    The failure is readable from the stream rather than raised out of it,
    because the consumer of these bytes is an HTTP client: an exception raised
    mid-iteration reaches the caller wrapped in a transport error, by which
    point "the camera went away" and "the worker refused this" are the same.
    """
    stream = await rtsp_relay_stream(str(h264_file))
    assert stream.failure is None

    await collect(stream)

    assert isinstance(stream.failure, CameraError)
    assert "ended without an error" in str(stream.failure)


@pytest.mark.asyncio
async def test_a_stream_is_iterable_once(h264_file):
    """Iterating twice would silently yield nothing, having closed the camera."""
    stream = await rtsp_relay_stream(str(h264_file))
    await collect(stream)

    with pytest.raises(RuntimeError, match="only be iterated once"):
        await collect(stream)


@pytest.mark.asyncio
async def test_closing_an_uniterated_stream_releases_the_camera(h264_file):
    """An upload refused before reading a byte must not leak the camera.

    Nothing else would close it: the cleanup that iteration does never runs.
    """
    stream = await rtsp_relay_stream(str(h264_file))

    await stream.aclose()

    # Idempotent, because a caller that closes and then hits the `finally`
    # that closes again is the ordinary shape of this.
    await stream.aclose()
    with pytest.raises(RuntimeError, match="already closed"):
        await collect(stream)


@pytest.mark.asyncio
async def test_an_unopenable_source_is_a_camera_error(tmp_path):
    """Raised at the call, before any upload is started with it."""
    with pytest.raises(CameraError, match="could not open"):
        await rtsp_relay_stream(str(tmp_path / "does-not-exist.mp4"))


@pytest.mark.asyncio
async def test_a_broken_mux_setup_is_a_mux_error_and_closes_the_camera(h264_file, monkeypatch):
    """A MuxError must not leave the camera open behind it."""
    real_open = av.open
    closed = []

    def open_failing_output(*args, **kwargs):
        if kwargs.get("mode") == "w":
            raise RuntimeError("no muxer for you")
        container = real_open(*args, **kwargs)

        # Recording the call rather than probing the container afterwards: a
        # closed PyAV container does not reliably raise, so "did we close it"
        # has to be observed where it happens. Wrapped rather than patched
        # because `close` is read-only on the C type.
        class ClosesAreRecorded:
            def __getattr__(self, name):
                return getattr(container, name)

            def close(self):
                closed.append(True)
                container.close()

        return ClosesAreRecorded()

    monkeypatch.setattr(av, "open", open_failing_output)

    with pytest.raises(MuxError, match="could not set up"):
        await rtsp_relay_stream(str(h264_file))

    assert closed == [True], "the input container was left open behind the MuxError"


@pytest.mark.asyncio
async def test_the_read_timeout_reaches_ffmpeg(h264_file, monkeypatch):
    """The parameter has to arrive as ffmpeg's option, in microseconds.

    A timeout that is accepted and not applied is the failure this guards: the
    demux blocks in C forever and no stop flag reaches it.
    """
    real_open = av.open
    seen = {}

    def record_options(*args, **kwargs):
        if kwargs.get("mode") != "w" and "options" in kwargs:
            seen.update(kwargs["options"])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", record_options)

    stream = await rtsp_relay_stream(str(h264_file), read_timeout_s=2.5)
    try:
        assert seen["timeout"] == str(2_500_000)
        assert seen["rtsp_transport"] == "tcp"
    finally:
        # Never iterated, so nothing else would release the camera.
        await stream.aclose()


@pytest.mark.asyncio
async def test_the_stream_is_an_async_iterable_of_bytes(h264_file):
    """What `upload_stream` accepts is what this has to be."""
    stream = await rtsp_relay_stream(str(h264_file))
    assert isinstance(stream, RtspRelayStream)

    chunks = stream.__aiter__()
    try:
        assert isinstance(await chunks.__anext__(), bytes)
    finally:
        # Abandoning it part-way is the caller-walks-away path: closing the
        # generator runs the shutdown that joins the muxing thread.
        await chunks.aclose()


@pytest.mark.asyncio
async def test_a_mux_failure_is_not_reported_as_a_camera_drop(h264_file, monkeypatch):
    """Muxing and demuxing raise the same type; they must not mean the same.

    `KlvRelay.relay()` calls `OutputContainer.mux()`, which raises
    `av.error.FFmpegError` exactly as a failing demux does. Recorded as a
    camera drop, a caller with reconnect enabled retries a defect forever
    instead of surfacing it.
    """
    from eyepop.relay.mux import KlvRelay

    def mux_fails(self, elapsed_s, packet):
        raise FFmpegError("mux refused the packet", "test")

    monkeypatch.setattr(KlvRelay, "relay", mux_fails)

    stream = await rtsp_relay_stream(str(h264_file))
    await collect(stream)

    assert isinstance(stream.failure, MuxError), stream.failure
    assert not isinstance(stream.failure, CameraError)
    assert "remuxing failed" in str(stream.failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [0, -1.0, float("nan"), float("inf")])
async def test_an_unusable_read_timeout_is_rejected(h264_file, invalid):
    """Rejected at the call rather than deep inside ffmpeg, or not at all.

    Zero or negative is not a shorter timeout - ffmpeg reads it as no socket
    timeout - which silently removes the only bound on a camera that stops
    sending without closing the connection. NaN and infinity would instead
    fail inconsistently inside int(), as ValueError and OverflowError.
    """
    with pytest.raises(ValueError, match="finite and positive"):
        await rtsp_relay_stream(str(h264_file), read_timeout_s=invalid)


@pytest.mark.asyncio
async def test_opening_the_camera_does_not_run_on_the_event_loop(h264_file, monkeypatch):
    """`av.open` talks to the camera and can block for the whole read timeout.

    Run on the event loop that stalls everything else on it, cancellation
    included. Asserting where it runs rather than how long it takes: a timing
    assertion would pass on a fast local file whether or not it was offloaded.
    """
    import threading

    real_open = av.open
    threads = []

    def record_thread(*args, **kwargs):
        threads.append(threading.current_thread())
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", record_thread)

    stream = await rtsp_relay_stream(str(h264_file))
    try:
        assert threads, "av.open was never called"
        assert all(t is not threading.main_thread() for t in threads), (
            "the camera was opened on the event loop thread"
        )
    finally:
        await stream.aclose()
