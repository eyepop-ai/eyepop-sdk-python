import asyncio
import io
import queue
from typing import AsyncGenerator

import av
import httpx

from eyepop.data.types.asset import Area
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_types import ComponentParams, MotionDetectConfig, VideoMode


async def relay_http_source(
        source_url: str,
        endpoint: WorkerEndpoint,
        params: list[ComponentParams] | None = None,
        motion_detect: MotionDetectConfig | None = None,
        roi: Area | None = None,
        fps: str | None = None
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
                fps=fps
            )
            while result := await job.predict():
                yield result

async def relay_rtsp_source(
        source_url: str,
        endpoint: WorkerEndpoint,
        params: list[ComponentParams] | None = None,
        motion_detect: MotionDetectConfig | None = None,
        roi: Area | None = None,
        fps: str | None = None
) -> AsyncGenerator[dict, None]:
    container = av.open(source_url, 'r', options={
        'rtsp_transport': 'tcp',
    })
    in_video_stream = container.streams.video[0]
    pipe = PipeBuffer()
    mpegts_muxer = av.open(pipe, format='mpegts', mode='w')
    out_video_stream = mpegts_muxer.add_stream_from_template(template=in_video_stream)

    def pipe_through():
        has_key_frame = False
        for packet in container.demux(in_video_stream):
            if packet.dts is None:
                continue
            if not has_key_frame:
                has_key_frame = packet.is_keyframe
            if not has_key_frame:
                continue
            packet.stream = out_video_stream
            mpegts_muxer.mux(packet)

    task = asyncio.create_task(asyncio.to_thread(pipe_through))

    job = await endpoint.upload_stream(
        pipe,
        mime_type="video/mpegts",
        is_live=True,
        video_mode=VideoMode.STREAM,
        # TODO extract first frame camera provided NTP timestamp and send as "captured_at_offset_ns"
        params=params,
        motion_detect=motion_detect,
        roi=roi,
        fps=fps,
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
