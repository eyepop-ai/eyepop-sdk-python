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

The relay starts uploading as soon as it has a stream to upload, rather than
waiting for the camera's first clock reference, so a frame that goes out ahead
of that reference carries no `captured_at`. A worker reading the camera
directly behaves the same way, waiting on the same sender report.

In practice that window is often empty. Opening the source costs about 2.4s
before the first frame is ever relayed - roughly 1.2s of stream probing, then
the wait for a keyframe - and the camera's clock reference usually arrives
inside it. Measured against a real Axis camera (AWSU-258), every prediction of
11807 carried a `captured_at` while the direct path spent its first 1.05s
without one. Those opening seconds are not analysed at all on the relay path,
rather than analysed without a capture time. A camera slower to send its first
sender report would shift that balance back.

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
from typing import AsyncGenerator

from eyepop.data.types.asset import Area
from eyepop.relay.rtsp import (
    INITIAL_BACKOFF_S,
    MAX_BACKOFF_S,
    READ_TIMEOUT_S,
    CameraError,
    MuxError,
    RelayError,
    UploadError,
    rtsp_relay_stream,
)
from eyepop.relay.st0601 import PlatformOrientation, SensorPosition
from eyepop.worker.camera import Camera
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import ComponentParams, MotionDetectConfig, VideoMode

log = logging.getLogger(__name__)

# Re-exported so copies of this file keep working unchanged: the muxing, the
# capture times and the error taxonomy all live in `eyepop.relay` now, and the
# only thing left here is the part that knows about a worker.
__all__ = [
    "INITIAL_BACKOFF_S",
    "MAX_BACKOFF_S",
    "READ_TIMEOUT_S",
    "CameraError",
    "MuxError",
    "RelayError",
    "UploadError",
    "relay_http_source",
    "relay_rtsp_source",
]


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

    A camera that stops sending counts as a drop even when it closes the
    connection politely, because that is what a drop actually looks like: the
    demux ends without an error rather than raising one. The consequence is
    that a *finite* RTSP source - a recording served over RTSP, rather than a
    camera - would be relayed again each time it ends. Relay one of those with
    ``reconnect=False``.

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

        # The session ended without the camera failing - the worker closed the
        # prediction stream, or the caller stopped consuming. A camera that
        # stops sending does not reach here: that is a CameraError above.
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
    """One RTSP session: relay its bytes to the worker and yield predictions.

    Everything about reading the camera and muxing its capture times lives in
    `eyepop.relay`. What is left here is the half that knows about a worker,
    which is the half you would replace to send the stream somewhere else.
    """
    stream = await rtsp_relay_stream(source_url, platform=platform, sensor=sensor)

    try:
        try:
            job = await endpoint.upload_stream(
                stream,
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
                # of stream. Report the camera failure the relay recorded rather
                # than the symptom the upload saw.
                if stream.failure:
                    raise stream.failure from error
                raise UploadError(f"prediction stream failed: {error}") from error
            if result is None:
                break
            yield result
    finally:
        # A no-op once the upload has read the stream to its end, which cleans
        # up as it goes. It matters when the upload was refused before reading
        # anything: the camera is open and nothing else would close it.
        await stream.aclose()

    # Raised after the generator body, so the caller sees why the stream ended
    # rather than an ordinary end of iteration. relay_rtsp_source turns a
    # CameraError into a reconnect.
    if stream.failure:
        raise stream.failure
