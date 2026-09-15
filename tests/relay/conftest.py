"""Fixtures shared by the relay tests."""

from __future__ import annotations

from fractions import Fraction

import av
import numpy as np
import pytest


@pytest.fixture
def h264_file(tmp_path):
    """A short H.264 file, so the muxing path runs for real.

    A file reaching its end drives the same code path a dropped camera does -
    measured in AWSU-258, severing an RTSP-over-TCP connection ends PyAV's
    demux normally rather than raising - which is what makes it usable here.
    """
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


@pytest.fixture
def h264_multi_gop_file(tmp_path):
    """Several groups of pictures, so a drop has somewhere to resume.

    `h264_file` holds one keyframe, which cannot show the property that matters
    when the relay sheds load: that it stops at a group boundary and starts
    again at the next one rather than punching a hole in a group.
    """
    path = tmp_path / "multi-gop.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=25)
    stream.width, stream.height = 160, 120
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": "ultrafast", "g": "10"}
    for index in range(120):
        image = np.full((120, 160, 3), index * 2 % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
        frame.pts = index
        frame.time_base = Fraction(1, 25)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path
