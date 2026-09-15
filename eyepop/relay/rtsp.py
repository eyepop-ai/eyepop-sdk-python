"""Relay an RTSP camera into MPEG-TS bytes carrying its own capture times.

What comes out is a byte stream, which is what an upload wants, so nothing here
knows about endpoints, jobs or predictions. That is deliberate: it keeps
``eyepop.relay`` free of any dependency on the client half of the SDK, and it
lets the bytes go wherever the caller wants rather than only to a worker.

**One call is one RTSP session.** A camera that drops ends the stream and
records why on the returned object; reconnecting is the caller's policy. That
split is not arbitrary - each reconnect has to become a *new upload*, because
one MPEG-TS cannot carry two RTSP sessions without renumbering the timestamps
the capture times depend on, and renumbering them is the one thing that would
break what this package exists to deliver.

``examples/relay_example.py`` shows the reconnect loop built on top.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from collections.abc import AsyncIterator

import av
from av.container import InputContainer, OutputContainer
from av.error import FFmpegError

from eyepop.relay.mux import KlvRelay, RelayStats
from eyepop.relay.pipe import PipeBuffer
from eyepop.relay.st0601 import PlatformOrientation, SensorPosition

__all__ = [
    "BackpressureError",
    "CameraError",
    "INITIAL_BACKOFF_S",
    "MAX_BACKOFF_S",
    "MAX_PENDING_BYTES",
    "MAX_STALL_S",
    "MuxError",
    "READ_TIMEOUT_S",
    "RelayError",
    "RtspRelayStream",
    "UploadError",
    "rtsp_relay_stream",
]

log = logging.getLogger(__name__)

#: How long to wait before the first reconnect attempt, and the ceiling the
#: backoff doubles towards. A camera rebooting takes tens of seconds, so
#: retrying faster than this only fills the log. Used by callers that implement
#: the reconnect policy; nothing here retries.
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

#: How long to wait for the camera to send something before treating the
#: session as dead. Long enough not to trip on a slow keyframe interval, short
#: enough that a camera pulled off the network is noticed rather than waited on.
READ_TIMEOUT_S = 10.0

#: How much to hand to the consumer at a time. The muxer's own writes are
#: smaller and irregular, so reading in chunks keeps the hand-off from becoming
#: one await per TS packet.
_CHUNK_BYTES = 65536

#: How far the upload may fall behind before the relay starts shedding frames.
#: Roughly thirty seconds of a 1 Mbps camera, eight of a 4 Mbps one - generous
#: enough that an ordinary network hiccup costs nothing, small enough that a
#: host running one session per camera does not grow without limit when several
#: stall at once.
MAX_PENDING_BYTES = 4 * 1024 * 1024

#: The fraction of :data:`MAX_PENDING_BYTES` the backlog has to fall back to
#: before relaying resumes. Resuming the moment it dips under the ceiling would
#: refill it on the next packet and shed a frame here and there across every
#: group of pictures, which is the one way of dropping that corrupts rather
#: than thins - so the gap is wide enough to be worth reopening.
_RESUME_FRACTION = 0.5

#: How long the relay may shed every frame before it gives up on the upload.
#: Dropping is meant to ride out a stall, not to replace the stream: past this
#: the upload is not slow, it is gone, and a new session recovers where waiting
#: does not. Counted only while the backlog is above the resume mark, so time
#: spent waiting for the next keyframe on a recovered upload does not end a
#: session that is about to be fine.
MAX_STALL_S = 15.0


class RelayError(Exception):
    """Base for the three ways a relay session can fail.

    Separated because the fix differs completely: the camera is yours to
    restart, the upload is a question for EyePop, and a mux failure is a bug
    here. A single opaque error leaves the user guessing which.
    """


class CameraError(RelayError):
    """The camera could not be opened or stopped delivering."""


class MuxError(RelayError):
    """The stream could not be remuxed. Unexpected - likely a defect."""


class UploadError(RelayError):
    """The worker rejected the stream or the connection to it failed.

    Never raised here - nothing in this module uploads anything. It lives here
    so that a caller catches one taxonomy rather than importing half of it from
    an example.

    Distinct from :class:`BackpressureError`, and the distinction is the point:
    this one means the worker will not take the stream, so retrying sends the
    same stream to the same refusal.
    """


class BackpressureError(RelayError):
    """The upload stayed too far behind the camera for too long.

    Its own type rather than an :class:`UploadError` because the two want
    opposite responses. An upload that was refused should not be retried; an
    upload that fell behind should, and a fresh session is how it recovers -
    it starts at a keyframe with its own timestamps, where the stalled one
    would have to push a backlog of stale video before catching up.
    """


class RtspRelayStream:
    """The MPEG-TS bytes of one RTSP session, and why the session ended.

    An ``AsyncIterable[bytes]``, so it goes straight into
    ``WorkerEndpoint.upload_stream()``. Iterate it once.

    The reason for being an object rather than a bare async generator is
    :attr:`failure`. A camera that drops has to reach the caller as a
    *camera* failure, and an error raised out of the byte stream reaches them
    wrapped in whatever the HTTP client made of it - by which point "the camera
    went away, reconnect" and "the worker refused this, do not" look the same.
    So the failure is recorded here and read afterwards.
    """

    def __init__(
        self,
        source_url: str,
        container: InputContainer,
        pipe: PipeBuffer,
        mpegts_muxer: OutputContainer,
        relay: KlvRelay,
        max_pending_bytes: int = MAX_PENDING_BYTES,
        max_stall_s: float = MAX_STALL_S,
    ) -> None:
        self._source_url = source_url
        self._container = container
        self._pipe = pipe
        self._mpegts_muxer = mpegts_muxer
        self._relay = relay
        self._max_pending_bytes = max_pending_bytes
        self._max_stall_s = max_stall_s
        # Rounded up, so a bound small enough that half of it is zero still
        # asks for some room to be made rather than resuming on an empty queue.
        self._resume_bytes = max(1, int(max_pending_bytes * _RESUME_FRACTION))
        self._stop = threading.Event()
        # Written by the muxing thread and read by the event loop once it has
        # finished. The thread cannot raise into the coroutine that started it.
        self._failure: list[RelayError] = []
        # Set when the demux ended of its own accord while the caller still
        # wanted frames. Kept apart from `_failure` so a concrete error wins.
        self._ended_at_eof: list[bool] = []
        self._eof_failure: RelayError | None = None
        self._started = False
        self._closed = False

    @property
    def stats(self) -> RelayStats:
        """What this session relayed, and what it shed.

        Live while the session runs - the muxing thread updates it in place -
        so a caller can watch ``dropped_packets`` climb rather than only learn
        about a stall once the session has ended.
        """
        return self._relay.stats

    @property
    def failure(self) -> RelayError | None:
        """Why the session ended, or ``None`` if it ended on its own terms.

        Readable as soon as the muxing thread records something, which is what
        lets a caller prefer it over the truncated-upload symptom it causes.
        """
        if self._failure:
            return self._failure[0]
        return self._eof_failure

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield the session's MPEG-TS bytes until the camera stops."""
        if self._started:
            raise RuntimeError("an RTSP relay stream can only be iterated once")
        if self._closed:
            raise RuntimeError("this RTSP relay stream is already closed")
        self._started = True

        task = asyncio.create_task(asyncio.to_thread(self._pipe_through))
        try:
            while True:
                # Blocking by design - the reader waits for the muxer - so it
                # goes to a thread rather than stalling the event loop.
                chunk = await asyncio.to_thread(self._pipe.read, _CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
        finally:
            # Reached on a clean end, on an error, and when the consumer stops
            # reading. The demux thread cannot be cancelled - it is blocked in
            # C - so it is asked to stop and then waited for.
            #
            # Order matters: the thread owns the container while it runs, and
            # closing it first pulls the input out from under a demux already
            # in progress, which hangs rather than returning.
            self._stop.set()
            await asyncio.gather(task, return_exceptions=True)
            self._container.close()
            self._closed = True

            # Last resort, once the thread has joined and had its say: a live
            # source that simply stopped has dropped. Only when nothing more
            # specific was recorded - a truncated upload is the better answer
            # when there is one.
            if not self._failure and self._ended_at_eof:
                self._eof_failure = CameraError(
                    f"stream from {self._source_url} ended without an error; "
                    f"a live source that stops sending has dropped"
                )

    async def aclose(self) -> None:
        """Release the camera for a stream that was never iterated.

        Iterating cleans up on its own, so this is for the paths that never
        got that far - an upload that was refused before it read a byte.
        """
        if self._started or self._closed:
            return
        self._closed = True
        try:
            self._mpegts_muxer.close()
        except Exception as error:
            log.warning("closing the MPEG-TS output failed: %s", error)
        self._container.close()

    def _pipe_through(self) -> None:
        started = time.monotonic()
        has_key_frame = False
        # Set while the relay is shedding packets, and separately while the
        # backlog is above the resume mark. The two come apart on the way back:
        # once the upload has caught up the relay is still dropping, because it
        # cannot resume until a keyframe, and that wait must not count as stall.
        dropping = False
        stalled_since: float | None = None
        try:
            for packet in self._container.demux(self._relay.in_video_stream):
                # Checked every packet rather than only on error: this is the
                # only way the thread ends early, and without it a camera that
                # keeps delivering keeps this thread alive after the caller has
                # stopped listening.
                if self._stop.is_set():
                    break
                if packet.dts is None:
                    continue

                # Measured before relaying rather than bounding the queue the
                # muxer writes into: a bounded queue blocks the write, and the
                # write happens inside a C callback this thread cannot be woken
                # out of. Shedding here instead keeps every write non-blocking,
                # and overshoots the bound by at most the one packet below.
                pending = self._pipe.pending_bytes
                if pending >= self._max_pending_bytes:
                    if stalled_since is None:
                        stalled_since = time.monotonic()
                    if not dropping:
                        dropping = True
                        # Redundant with the resume condition below, which only
                        # fires on a keyframe, and kept because it states the
                        # invariant: nothing relayed after a gap until an IDR.
                        has_key_frame = False
                        self._relay.note_dropped(new_episode=True)
                        log.warning(
                            "upload of %s is %d bytes behind; dropping until it catches up",
                            self._source_url, pending,
                        )
                        continue
                elif pending <= self._resume_bytes:
                    stalled_since = None

                if dropping:
                    if stalled_since is None and packet.is_keyframe:
                        dropping = False
                        log.info(
                            "upload of %s caught up; resuming (%d packets dropped so far)",
                            self._source_url, self._relay.stats.dropped_packets,
                        )
                    elif (
                        stalled_since is not None
                        and time.monotonic() - stalled_since >= self._max_stall_s
                    ):
                        # Ending the session beats dropping forever. A relay
                        # that sheds every frame is alive and useless, and only
                        # a new session recovers: this one would have to push a
                        # backlog of stale video before it caught up.
                        self._failure.append(BackpressureError(
                            f"upload of {self._source_url} stayed more than "
                            f"{self._resume_bytes} bytes behind for "
                            f"{self._max_stall_s:.0f}s"
                        ))
                        break

                if dropping:
                    self._relay.note_dropped()
                    continue

                if not has_key_frame:
                    has_key_frame = packet.is_keyframe
                if not has_key_frame:
                    continue
                # Relaying starts now, not once a capture time is available.
                # The leading frames go out unstamped, which is what the direct
                # RTSP path does too while it waits for its first sender report.
                try:
                    self._relay.relay(time.monotonic() - started, packet)
                except FFmpegError as error:
                    # Scoped tightly to the muxing call, because demux and mux
                    # raise the same type and the handler below cannot tell
                    # them apart. Without this a mux defect is recorded as a
                    # camera drop, and a caller with reconnect enabled retries
                    # a bug forever instead of surfacing it.
                    self._failure.append(MuxError(f"remuxing failed: {error}"))
                    break
            else:
                # The demux ended without raising. On a live camera that is a
                # drop, not a stream finishing: measured against a real camera
                # (AWSU-258), severing an RTSP-over-TCP connection ends the
                # demux generator *normally* and raises nothing at all.
                #
                # Without this, a dropped camera is indistinguishable from a
                # finite file running out and a caller's reconnect never fires.
                #
                # Recorded as a flag rather than straight into `_failure`: this
                # is an inference from silence, so anything concrete - a demux
                # error, or a close() that failed and truncated the upload -
                # must outrank it. It becomes a CameraError once the thread has
                # joined, and only if nothing better was found.
                #
                # Guarded on `_stop` so a caller walking away stays a clean
                # shutdown rather than an error.
                if not self._stop.is_set():
                    self._ended_at_eof.append(True)
        except FFmpegError as error:
            # The camera going away arrives here as a demux error. It is the
            # expected end of a live session, not a defect.
            self._failure.append(
                CameraError(f"stream from {self._source_url} ended: {error}")
            )
        except Exception as error:
            self._failure.append(MuxError(f"remuxing failed: {error}"))
        finally:
            # A camera that disconnects, a finite source that ends, or anything
            # raised above all land here. Without it the reader blocks forever
            # on an empty queue and the consumer never sees the end of stream.
            try:
                self._mpegts_muxer.close()
            except Exception as error:
                # close() flushes what is still buffered, so a failure here
                # means the stream is truncated. EOF is signalled either way -
                # the reader must not be left blocked - and without recording
                # this the truncated stream ends and reads as a clean finish.
                log.warning("closing the MPEG-TS output failed: %s", error)
                if not self._failure:
                    # Never over an earlier failure: a camera that dropped is
                    # why the close failed, and it is the more useful answer.
                    self._failure.append(
                        MuxError(f"closing the MPEG-TS output failed: {error}")
                    )
            self._pipe.signal_eof()


async def rtsp_relay_stream(
    source_url: str,
    platform: PlatformOrientation | None = None,
    sensor: SensorPosition | None = None,
    read_timeout_s: float = READ_TIMEOUT_S,
    max_pending_bytes: int = MAX_PENDING_BYTES,
    max_stall_s: float = MAX_STALL_S,
) -> RtspRelayStream:
    """Open an RTSP camera and return its stream as MPEG-TS bytes.

    The video is copied packet for packet - nothing is decoded and nothing is
    re-encoded - and the camera's own capture times ride alongside it as MISB
    ST 0601 KLV, which is what lets a worker report ``captured_at`` for a
    camera it cannot reach itself.

    Pass the result straight to ``WorkerEndpoint.upload_stream()``, or anywhere
    else that takes an ``AsyncIterable[bytes]``. Check
    :attr:`RtspRelayStream.failure` once the stream ends to find out whether it
    ended or broke.

    Awaitable because opening talks to the camera and can take up to
    ``read_timeout_s``: the work happens on a worker thread so a camera that is
    slow to answer - or not there at all - does not stall the event loop and
    block cancellation along with it.

    An upload slower than the camera is shed rather than buffered. Once more
    than ``max_pending_bytes`` is waiting to go out, whole groups of pictures
    are dropped until the upload catches up and the next keyframe arrives - the
    stream thins and stays current instead of growing without limit and falling
    permanently behind real time. Surviving frames keep their capture times:
    an anchor is built from the packet it describes, so a dropped packet takes
    its anchor with it and nothing is left pointing at a frame that never went.
    Watch :attr:`RtspRelayStream.stats` to see what it cost.

    Dropping is for riding out a stall, not for replacing the stream. An upload
    that stays behind for ``max_stall_s`` ends the session with a
    ``BackpressureError``, which a caller should treat as a reconnect.

    Raises ``CameraError`` if the camera cannot be opened, ``MuxError`` if the
    MPEG-TS output cannot be set up, and ``ValueError`` for an unusable
    ``read_timeout_s``, ``max_pending_bytes`` or ``max_stall_s``.
    """
    # A timeout of zero or less is not a shorter timeout: ffmpeg reads it as no
    # socket timeout at all, which removes the only bound on a camera that
    # stops sending without closing the connection. NaN and infinity would
    # otherwise fail later and inconsistently, inside int().
    if not math.isfinite(read_timeout_s) or read_timeout_s <= 0:
        raise ValueError(
            f"read_timeout_s must be finite and positive, got {read_timeout_s!r}"
        )

    # A bound of zero or less would drop every packet forever, and a stall
    # window of zero would end the session the first time a chunk was in
    # flight. Both are caught here rather than surfacing as a relay that runs
    # and produces nothing.
    if max_pending_bytes <= 0:
        raise ValueError(
            f"max_pending_bytes must be positive, got {max_pending_bytes!r}"
        )
    if not math.isfinite(max_stall_s) or max_stall_s <= 0:
        raise ValueError(
            f"max_stall_s must be finite and positive, got {max_stall_s!r}"
        )

    return await asyncio.to_thread(
        _open_stream, source_url, platform, sensor, read_timeout_s,
        max_pending_bytes, max_stall_s,
    )


def _open_stream(
    source_url: str,
    platform: PlatformOrientation | None,
    sensor: SensorPosition | None,
    read_timeout_s: float,
    max_pending_bytes: int,
    max_stall_s: float,
) -> RtspRelayStream:
    """The blocking half of opening a camera, for a worker thread."""
    try:
        # TCP to match the direct path: gst-ep-source forces protocols=TCP
        # there, and the two have to see the same stream for their timestamps
        # to compare.
        container = av.open(source_url, "r", options={
            "rtsp_transport": "tcp",
            # Without a read timeout a camera that stops sending without
            # closing the connection - powered off, cable pulled - leaves the
            # demux blocked in C forever, which no stop flag can reach. With
            # one it surfaces as a demux error, which the caller can retry.
            "timeout": str(int(read_timeout_s * 1_000_000)),
        })
    except FFmpegError as error:
        raise CameraError(f"could not open {source_url}: {error}") from error

    pipe = PipeBuffer()
    try:
        mpegts_muxer = av.open(pipe, format="mpegts", mode="w", options={
            # How long FFmpeg may hold a packet waiting for the other stream to
            # catch up, so that it can interleave the two. The default is ten
            # seconds, and on a camera that sends no capture times the KLV
            # stream produces nothing to interleave against - so ten seconds of
            # video accumulates inside FFmpeg, where the backlog the relay
            # measures cannot see it and the bound above never fires. Nothing
            # here needs the interleaving: an anchor is muxed immediately after
            # the packet it describes and carries that packet's timestamps, so
            # the order is already right when it arrives.
            #
            # Not zero: FFmpeg reads zero as unlimited, which is the opposite.
            "max_interleave_delta": "1000000",
        })
        relay = KlvRelay(container, mpegts_muxer, platform=platform, sensor=sensor)
    except Exception as error:
        container.close()
        raise MuxError(f"could not set up the MPEG-TS output: {error}") from error

    return RtspRelayStream(
        source_url, container, pipe, mpegts_muxer, relay,
        max_pending_bytes=max_pending_bytes, max_stall_s=max_stall_s,
    )
