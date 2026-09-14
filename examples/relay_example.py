import asyncio
import io
import queue
import time
from typing import AsyncGenerator

import av
import httpx

from eyepop.data.types.asset import Area
from eyepop.relay.mux import KlvRelay
from eyepop.relay.st0601 import PlatformOrientation, SensorPosition
from eyepop.worker.camera import Camera
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import ComponentParams, MotionDetectConfig, VideoMode


async def relay_http_source(
        source_url: str,
        endpoint: WorkerEndpoint,
        params: list[ComponentParams] | None = None,
        motion_detect: MotionDetectConfig | None = None,
        roi: Area | None = None,
        fps: str | None = None,
        camera: Camera | None = None
) -> AsyncGenerator[dict, None]:
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
) -> AsyncGenerator[dict, None]:
    # TCP to match the direct path: gst-ep-source forces protocols=TCP there,
    # and the two have to see the same stream for their timestamps to compare.
    container = av.open(source_url, 'r', options={
        'rtsp_transport': 'tcp',
    })
    pipe = PipeBuffer()
    mpegts_muxer = av.open(pipe, format='mpegts', mode='w')
    relay = KlvRelay(container, mpegts_muxer, platform=platform, sensor=sensor)

    def pipe_through():
        started = time.monotonic()
        has_key_frame = False
        for packet in container.demux(relay.in_video_stream):
            if packet.dts is None:
                continue
            if not has_key_frame:
                has_key_frame = packet.is_keyframe
            if not has_key_frame:
                continue
            # Uploading starts now, not once a capture time is available. The
            # leading frames go out unstamped, which is what the direct RTSP
            # path does too while it waits for its first sender report.
            relay.relay(time.monotonic() - started, packet)

    task = asyncio.create_task(asyncio.to_thread(pipe_through))

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
    while result := await job.predict():
        yield result

    await asyncio.gather(task)


class PipeBuffer(io.RawIOBase):
    def __init__(self):
        self.queue = queue.Queue()
        self.buffer = b""

    def writable(self):
        return True

    def write(self, b):
        if isinstance(b, str):
            b = b.encode('utf-8')
        self.queue.put(b)
        return len(b)

    def read(self, n=-1):
        # Fetch chunks from queue if our internal buffer is empty
        if not self.buffer:
            try:
                # Blocks until data is available
                self.buffer = self.queue.get(block=True, timeout=None)
            except queue.Empty:
                return b""  # EOF

        # If n is negative, read everything available
        if n < 0:
            res, self.buffer = self.buffer, b""
            return res

        # Otherwise, slice out the exact number of bytes requested
        res = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return res
