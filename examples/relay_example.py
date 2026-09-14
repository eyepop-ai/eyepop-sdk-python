"""Relay a camera that is not reachable from the internet into EyePop.

The reference implementation for forwarding a private RTSP camera. Your
application reads the camera on your own network and forwards the stream to a
worker, and the predictions come back carrying `captured_at` - the moment the
camera captured the frame - as if the worker had read the camera itself.

Two things make that true, and both are worth understanding before copying this:

**The stream is remuxed, never re-encoded.** Packets are copied from the RTSP
input into an MPEG-TS output untouched. No decoder and no encoder run here, so
the relay costs almost nothing and loses no quality. It is also why the video
timestamps survive: the worker matches each capture time to a frame by exact
presentation timestamp, and re-encoding would renumber them.

**The capture time travels in-band as MISB ST 0601 KLV.** A worker reading RTSP
directly gets the camera's clock from RTCP sender reports. Those do not survive
the hop into an HTTP upload, so the relay reads the camera's reference itself
and writes it into a metadata track alongside the video. The worker parses that
track and attaches the capture time to the decoded frame, arriving at exactly
the value the direct path would have produced.

## `captured_at` is missing at first, and that is expected

The relay starts uploading immediately rather than waiting for the camera's
first clock reference, so the first few seconds of predictions carry no
`captured_at`. A worker reading the camera directly behaves the same way - it
is waiting for the same sender report - and measured against a real camera the
window is around 2 seconds.

A reconnect starts a new RTSP session, so a fresh window follows each one. The
relay does not extrapolate across the gap: a timestamp that is missing is
better than one that is invented.

This is different from a camera that has no usable clock at all, which produces
no `captured_at` ever. `CaptureClock` warns about that case separately.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import AsyncGenerator

import av

from eyepop.data.types.asset import Area
from eyepop.relay.mux import KlvRelay
from eyepop.relay.pipe import PipeBuffer
from eyepop.relay.st0601 import PlatformOrientation, SensorPosition
from eyepop.worker.camera import Camera
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import ComponentParams, MotionDetectConfig, VideoMode

log = logging.getLogger(__name__)

#: How long to wait before the first reconnect attempt, and the ceiling the
#: backoff doubles towards. A camera rebooting takes tens of seconds, so
#: retrying faster than this only fills the log.
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 30.0

#: How long to wait for the camera to send something before treating the
#: session as dead. Long enough not to trip on a slow keyframe interval, short
#: enough that a camera pulled off the network is noticed rather than waited on.
READ_TIMEOUT_S = 10.0


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
    """The worker rejected the stream or the connection to it failed."""


async def relay_http_source(
        source_url: str,
        endpoint: WorkerEndpoint,
        params: list[ComponentParams] | None = None,
        motion_detect: MotionDetectConfig | None = None,
        roi: Area | None = None,
        fps: str | None = None,
        camera: Camera | None = None
) -> AsyncGenerator[dict, None]:
    # Imported here rather than at module scope: it is the only thing in this
    # file that needs httpx, and the RTSP relay below - which is what this
    # example is about - should not fail to import without it.
    import httpx

    async with httpx.AsyncClient() as http_client:
        async with http_client.stream("GET", source_url) as response:
            response.raise_for_status()
            job = await endpoint.upload_stream(
                response.aiter_bytes(),
                mime_type=response.headers.get("content-type"),
                params=params,
                motion_detect=motion_detect,
                roi=roi,
                fps=fps,
                camera=camera
            )
            while result := await job.predict():
                yield result


async def relay_rtsp_source(
        source_url: str,
        endpoint: WorkerEndpoint,
        params: list[ComponentParams] | None = None,
        motion_detect: MotionDetectConfig | None = None,
        roi: Area | None = None,
        fps: str | None = None,
        camera: Camera | None = None,
        platform: PlatformOrientation | None = None,
        sensor: SensorPosition | None = None,
        reconnect: bool = True,
        max_backoff_s: float = MAX_BACKOFF_S,
) -> AsyncGenerator[dict, None]:
    """Relay an RTSP camera, yielding predictions until the source ends.

    A camera on your premises will drop - rebooted, unplugged, a switch
    restarted - so by default the relay reconnects and keeps going, and the
    predictions continue from the caller's point of view. Pass
    ``reconnect=False`` to let a camera failure surface instead.

    Each reconnect is a new RTSP session and therefore a new upload: the
    camera's timestamps restart, and one MPEG-TS stream cannot carry two
    sessions without renumbering them, which is the one thing that would break
    the capture times this relay exists to deliver.
    """
    if not math.isfinite(max_backoff_s) or max_backoff_s < 0:
        raise ValueError(f"max_backoff_s must be finite and non-negative, got {max_backoff_s!r}")

    # A ceiling below the opening delay would otherwise be ignored for the
    # first retry and again after every success, so the cap is applied to the
    # starting value rather than only to the doubling.
    initial_backoff = min(INITIAL_BACKOFF_S, max_backoff_s)
    backoff = initial_backoff

    while True:
        try:
            async for result in _relay_one_session(
                    source_url, endpoint,
                    params=params, motion_detect=motion_detect, roi=roi, fps=fps,
                    camera=camera, platform=platform, sensor=sensor,
            ):
                # This runs only when a session yields, so it is already a
                # reset-on-success: a camera that accepts the connection and
                # then fails without delivering anything never reaches here and
                # keeps backing off instead of being retried in a tight loop.
                backoff = initial_backoff
                yield result
        except CameraError as error:
            if not reconnect:
                raise
            log.warning("camera %s: %s - reconnecting in %.1fs", source_url, error, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff_s)
            continue

        # The source ended on its own terms rather than failing. A finite
        # stream is done; reconnecting to it would replay it forever.
        return


async def _relay_one_session(
        source_url: str,
        endpoint: WorkerEndpoint,
        params: list[ComponentParams] | None = None,
        motion_detect: MotionDetectConfig | None = None,
        roi: Area | None = None,
        fps: str | None = None,
        camera: Camera | None = None,
        platform: PlatformOrientation | None = None,
        sensor: SensorPosition | None = None,
) -> AsyncGenerator[dict, None]:
    """One RTSP session, from opening the camera to the end of its stream."""
    try:
        # TCP to match the direct path: gst-ep-source forces protocols=TCP
        # there, and the two have to see the same stream for their timestamps
        # to compare.
        container = av.open(source_url, 'r', options={
            'rtsp_transport': 'tcp',
            # Without a read timeout a camera that stops sending without
            # closing the connection - powered off, cable pulled - leaves the
            # demux blocked in C forever, which no stop flag can reach. With
            # one it surfaces as a demux error, which is a reconnect.
            'timeout': str(int(READ_TIMEOUT_S * 1_000_000)),
        })
    except av.FFmpegError as error:
        raise CameraError(f"could not open {source_url}: {error}") from error

    pipe = PipeBuffer()
    stop = threading.Event()
    # Set by the muxing thread and read by this one after it finishes. The
    # thread cannot raise into the coroutine that started it.
    failure: list[RelayError] = []

    try:
        mpegts_muxer = av.open(pipe, format='mpegts', mode='w')
        relay = KlvRelay(container, mpegts_muxer, platform=platform, sensor=sensor)
    except Exception as error:
        container.close()
        raise MuxError(f"could not set up the MPEG-TS output: {error}") from error

    def pipe_through():
        started = time.monotonic()
        has_key_frame = False
        try:
            for packet in container.demux(relay.in_video_stream):
                # Checked every packet rather than only on error: this is the
                # only way the thread ends early, and without it a camera that
                # keeps delivering keeps this thread alive after the caller has
                # stopped listening.
                if stop.is_set():
                    break
                if packet.dts is None:
                    continue
                if not has_key_frame:
                    has_key_frame = packet.is_keyframe
                if not has_key_frame:
                    continue
                # Uploading starts now, not once a capture time is available.
                # The leading frames go out unstamped, which is what the direct
                # RTSP path does too while it waits for its first sender report.
                relay.relay(time.monotonic() - started, packet)
        except av.FFmpegError as error:
            # The camera going away arrives here as a demux error. It is the
            # expected end of a live session, not a defect.
            failure.append(CameraError(f"stream from {source_url} ended: {error}"))
        except Exception as error:
            failure.append(MuxError(f"remuxing failed: {error}"))
        finally:
            # A camera that disconnects, a finite source that ends, or anything
            # raised above all land here. Without it the reader blocks forever
            # on an empty queue and the upload never sees the end of the stream.
            try:
                mpegts_muxer.close()
            except Exception as error:
                # close() flushes what is still buffered, so a failure here
                # means the upload is truncated. EOF is signalled either way -
                # the reader must not be left blocked - and without recording
                # this the truncated stream ends and reads as a clean finish.
                log.warning("closing the MPEG-TS output failed: %s", error)
                if not failure:
                    # Never over an earlier failure: a camera that dropped is
                    # why the close failed, and it is the more useful answer.
                    failure.append(MuxError(f"closing the MPEG-TS output failed: {error}"))
            pipe.signal_eof()

    task = asyncio.create_task(asyncio.to_thread(pipe_through))

    try:
        try:
            job = await endpoint.upload_stream(
                pipe,
                mime_type="video/mpegts",
                is_live=True,
                video_mode=VideoMode.STREAM,
                params=params,
                motion_detect=motion_detect,
                roi=roi,
                fps=fps,
                camera=camera,
            )
        except Exception as error:
            raise UploadError(f"the worker did not accept the stream: {error}") from error

        while True:
            try:
                result = await job.predict()
            except Exception as error:
                # A camera failure shows up here too, as the upload running out
                # of stream. Report the camera failure the thread recorded
                # rather than the symptom the upload saw.
                if failure:
                    raise failure[0] from error
                raise UploadError(f"prediction stream failed: {error}") from error
            if result is None:
                break
            yield result
    finally:
        # Reached on a clean end, on an error, and when the caller stops
        # consuming the generator. The demux thread cannot be cancelled - it is
        # blocked in C - so it is asked to stop and then waited for.
        #
        # Order matters: the thread owns the container while it runs, and
        # closing it first pulls the input out from under a demux already in
        # progress, which hangs rather than returning.
        stop.set()
        await asyncio.gather(task, return_exceptions=True)
        container.close()

    # Raised after the generator body, so the caller sees why the stream ended
    # rather than an ordinary end of iteration. relay_rtsp_source turns a
    # CameraError into a reconnect.
    if failure:
        raise failure[0]
