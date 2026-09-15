"""`eyepop.relay.rtsp` on its own terms - no endpoint, no worker, no upload.

Being able to write this file is the point of the refactor: the relay used to
be reachable only through something shaped like a `WorkerEndpoint`, so every
test of the muxing had to carry a fake one. Here the stream is just bytes.
"""

from __future__ import annotations

import av
import pytest

from eyepop.relay import (
    CameraError,
    MuxError,
    RtspRelayStream,
    create_rtsp_relay_stream,
)


async def collect(stream) -> bytes:
    return b"".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_it_yields_mpegts_bytes(h264_file):
    """The bytes are a real MPEG-TS, not an opaque blob."""
    stream = create_rtsp_relay_stream(str(h264_file))
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
    stream = create_rtsp_relay_stream(str(h264_file))
    assert stream.failure is None

    await collect(stream)

    assert isinstance(stream.failure, CameraError)
    assert "ended without an error" in str(stream.failure)


@pytest.mark.asyncio
async def test_a_stream_is_iterable_once(h264_file):
    """Iterating twice would silently yield nothing, having closed the camera."""
    stream = create_rtsp_relay_stream(str(h264_file))
    await collect(stream)

    with pytest.raises(RuntimeError, match="only be iterated once"):
        await collect(stream)


@pytest.mark.asyncio
async def test_closing_an_uniterated_stream_releases_the_camera(h264_file):
    """An upload refused before reading a byte must not leak the camera.

    Nothing else would close it: the cleanup that iteration does never runs.
    """
    stream = create_rtsp_relay_stream(str(h264_file))

    await stream.aclose()

    # Idempotent, because a caller that closes and then hits the `finally`
    # that closes again is the ordinary shape of this.
    await stream.aclose()
    with pytest.raises(RuntimeError, match="already closed"):
        await collect(stream)


def test_an_unopenable_source_is_a_camera_error(tmp_path):
    """Raised at the call, before any upload is started with it."""
    with pytest.raises(CameraError, match="could not open"):
        create_rtsp_relay_stream(str(tmp_path / "does-not-exist.mp4"))


def test_a_broken_mux_setup_is_a_mux_error_and_closes_the_camera(h264_file, monkeypatch):
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
        create_rtsp_relay_stream(str(h264_file))

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

    stream = create_rtsp_relay_stream(str(h264_file), read_timeout_s=2.5)
    try:
        assert seen["timeout"] == str(2_500_000)
        assert seen["rtsp_transport"] == "tcp"
    finally:
        # Never iterated, so nothing else would release the camera.
        await stream.aclose()


@pytest.mark.asyncio
async def test_the_stream_is_an_async_iterable_of_bytes(h264_file):
    """What `upload_stream` accepts is what this has to be."""
    stream = create_rtsp_relay_stream(str(h264_file))
    assert isinstance(stream, RtspRelayStream)

    chunks = stream.__aiter__()
    try:
        assert isinstance(await chunks.__anext__(), bytes)
    finally:
        # Abandoning it part-way is the caller-walks-away path: closing the
        # generator runs the shutdown that joins the muxing thread.
        await chunks.aclose()
