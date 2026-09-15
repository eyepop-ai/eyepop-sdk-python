"""The relay's output, checked against the bytes in the transport stream.

Runs against a generated H.264 file rather than an RTSP camera, with the side
data injected through the seam KlvRelay exposes for exactly this. A test that
needs a camera is a test that does not run.
"""

from __future__ import annotations

import io
import struct
from fractions import Fraction

import av
import numpy as np
import pytest

from eyepop.relay.mux import KlvRelay
from eyepop.relay.ntp import CaptureClock
from eyepop.relay.st0601 import UAS_LDS_UNIVERSAL_LABEL
from tests.relay.tsparse import klv_payloads, stream_ids

FPS = 25
FRAME_COUNT = 60
BASE_CAPTURE_US = 1789316993000000
FRAME_US = 1_000_000 // FPS


@pytest.fixture
def h264_source(tmp_path):
    """An H.264 file with B-frames, so PTS is not monotonic.

    x264 leaves B-frames off under several presets; they are forced on because a
    source without reordered PTS lets a DTS bug through. Stamping the KLV
    packet's DTS from the video packet's PTS fails only once PTS and DTS
    diverge, and then partway into the stream rather than on the first packet.
    """
    path = tmp_path / "source.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height = 320, 240
    stream.pix_fmt = "yuv420p"
    stream.options = {"bf": "2", "b-adapt": "0", "preset": "ultrafast", "g": "25"}

    rng = np.random.default_rng(seed=7)
    for index in range(FRAME_COUNT):
        # Moving content, so the encoder has something to predict and the
        # reordering the fixture exists to produce actually happens.
        image = np.full((240, 320, 3), index * 4 % 256, dtype=np.uint8)
        image[:, (index * 5) % 320 : ((index * 5) % 320) + 20] = rng.integers(
            0, 255, (240, min(20, 320 - (index * 5) % 320), 3), dtype=np.uint8
        )
        frame = av.VideoFrame.from_ndarray(image, format="rgb24").reformat(format="yuv420p")
        frame.pts = index
        frame.time_base = Fraction(1, FPS)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def prft(unix_us: int) -> bytes:
    return struct.pack("=qi", unix_us, 0)


def relay_to_bytes(source_path, *, stamp_from: int = 0, clock: CaptureClock | None = None):
    """Relay the file into an in-memory MPEG-TS, stamping from a given frame."""
    buffer = io.BytesIO()
    source = av.open(str(source_path))
    output = av.open(buffer, mode="w", format="mpegts")

    counter = {"n": 0}

    def side_data(_packet):
        index = counter["n"]
        counter["n"] += 1
        if index < stamp_from:
            return None, None
        return prft(BASE_CAPTURE_US + index * FRAME_US), None

    relay = KlvRelay(source, output, clock=clock, side_data=side_data)
    for index, packet in enumerate(source.demux(relay.in_video_stream)):
        if packet.dts is None:
            continue
        relay.relay(index / FPS, packet)
    output.close()
    source.close()
    return buffer.getvalue(), relay


def test_the_source_really_has_reordered_pts(h264_source):
    """Otherwise every DTS assertion below is vacuous."""
    container = av.open(str(h264_source))
    stream = container.streams.video[0]
    reordered = sum(
        1 for p in container.demux(stream) if p.dts is not None and p.pts != p.dts
    )
    container.close()
    assert reordered > 0, "no B-frames: this fixture cannot catch a PTS/DTS mix-up"


def test_output_carries_one_klv_set_per_frame(h264_source):
    data, relay = relay_to_bytes(h264_source)
    payloads = klv_payloads(data, UAS_LDS_UNIVERSAL_LABEL)
    assert relay.stats.klv_packets == relay.stats.video_packets
    assert len(payloads) == relay.stats.klv_packets


def test_timestamps_round_trip_byte_for_byte(h264_source):
    """What the worker parses back must equal what was written."""
    data, _ = relay_to_bytes(h264_source)
    payloads = klv_payloads(data, UAS_LDS_UNIVERSAL_LABEL)
    assert payloads

    written = set()
    for payload in payloads:
        assert payload[17:19] == bytes((2, 8))
        (value,) = struct.unpack(">Q", payload[19:27])
        written.add(value)

    expected = {BASE_CAPTURE_US + i * FRAME_US for i in range(len(payloads))}
    assert written == expected


def test_the_universal_label_survives_muxing_intact(h264_source):
    """FFmpeg's own demuxer drops five bytes of this; the muxer does not."""
    data, _ = relay_to_bytes(h264_source)
    for payload in klv_payloads(data, UAS_LDS_UNIVERSAL_LABEL):
        assert payload[:16] == UAS_LDS_UNIVERSAL_LABEL
        assert len(payload) == 34


def test_klv_rides_on_the_metadata_stream_id(h264_source):
    data, _ = relay_to_bytes(h264_source)
    assert 0xFC in stream_ids(data), "KLV should be PES stream_id 0xFC"


def test_video_is_copied_not_decoded(h264_source):
    """Passthrough, asserted rather than inferred from the output playing.

    A relay that decoded and re-encoded would still produce a valid stream, and
    would still pass every other test here.
    """
    source = av.open(str(h264_source))
    output = av.open(io.BytesIO(), mode="w", format="mpegts")
    relay = KlvRelay(source, output, side_data=lambda _p: (prft(BASE_CAPTURE_US), None))

    for index, packet in enumerate(source.demux(relay.in_video_stream)):
        if packet.dts is None:
            continue
        relay.relay(index / FPS, packet)
        assert packet.stream.codec_context is not relay.in_video_stream.codec_context

    # A data stream has no codec context at all; that is correct, not a failure.
    assert relay._out_klv.codec_context is None
    output.close()
    source.close()


def stream_timestamps(data: bytes, kind: str, field: str = "pts") -> list[int]:
    """One stream's timestamps, read back out of the bytes."""
    container = av.open(io.BytesIO(data), format="mpegts")
    try:
        stream = next(s for s in container.streams if s.type == kind)
        values = (getattr(p, field) for p in container.demux(stream))
        return [value for value in values if value is not None]
    finally:
        container.close()


def test_klv_dts_is_monotonic_even_though_pts_is_not(h264_source):
    """The trap that fails partway into a stream rather than at its start."""
    data, _ = relay_to_bytes(h264_source)
    container = av.open(io.BytesIO(data), format="mpegts")
    klv_stream = next(s for s in container.streams if s.type == "data")

    dts_values = [p.dts for p in container.demux(klv_stream) if p.dts is not None]
    container.close()

    assert dts_values
    assert dts_values == sorted(dts_values)
    # Monotonicity alone is too weak to be worth asserting on its own: it
    # survives any uniform rescaling of the timestamps, so it held while every
    # anchor after the first pointed at the wrong frame (SDK-21). Tying the
    # values to the video stream's own DTS is what makes this test able to
    # fail. Against PTS it could not: this fixture has B-frames, which is the
    # whole reason DTS is carried separately.
    assert sorted(dts_values) == sorted(stream_timestamps(data, "video", "dts"))


def test_anchor_timestamps_match_their_frames_exactly(h264_source):
    """Each anchor lands on a frame that is in the stream, not near one.

    The whole basis of matching an anchor to a frame on the far side. An anchor
    that misses attaches its capture time to whichever frame is nearest, so the
    value looks plausible and is wrong - worse than no value at all.
    """
    data, relay = relay_to_bytes(h264_source)

    video_pts = sorted(stream_timestamps(data, "video"))
    klv_pts = sorted(stream_timestamps(data, "data"))

    assert len(klv_pts) == relay.stats.klv_packets
    assert klv_pts == video_pts


def test_anchors_align_when_the_source_time_base_is_not_90khz(h264_source):
    """The regression guard for SDK-21.

    MPEG-TS always writes 90 kHz, so a source that reports anything else has
    its timestamps rescaled on the way out. Reading a video packet's PTS after
    muxing it - which rescales in place - and then labelling the anchor with
    the source time base scaled them a second time, by exactly
    90000/source. On a live RTSP source that factor is 1, which is why the
    defect was invisible on the only path anyone runs.

    This fixture is an mp4 and reports 1/12800, so the factor here is 7.03125
    and a regression shows up immediately.
    """
    container = av.open(str(h264_source))
    source_time_base = container.streams.video[0].time_base
    container.close()
    assert source_time_base != Fraction(1, 90000), (
        "this test is only meaningful on a source that is not already 90 kHz"
    )

    data, _ = relay_to_bytes(h264_source)

    video_pts = sorted(stream_timestamps(data, "video"))
    klv_pts = sorted(stream_timestamps(data, "data"))

    assert klv_pts == video_pts, (
        f"anchors at {sorted(set(klv_pts) - set(video_pts))[:5]} describe frames "
        f"that are not in the stream"
    )


def test_frames_before_the_first_anchor_are_relayed_unstamped(h264_source):
    """The relay does not wait for an anchor, and does not invent one either."""
    data, relay = relay_to_bytes(h264_source, stamp_from=10)

    assert relay.stats.frames_before_first_anchor == 10
    assert relay.stats.video_packets > relay.stats.klv_packets
    assert len(klv_payloads(data, UAS_LDS_UNIVERSAL_LABEL)) == relay.stats.klv_packets


def test_a_source_with_no_anchors_at_all_still_relays_video(h264_source):
    data, relay = relay_to_bytes(h264_source, stamp_from=FRAME_COUNT * 2)

    assert relay.stats.video_packets > 0
    assert relay.stats.klv_packets == 0
    assert klv_payloads(data, UAS_LDS_UNIVERSAL_LABEL) == []


def test_only_the_startup_window_counts_as_before_the_first_anchor(h264_source):
    """Picture timing arrives on roughly half of packets on a real camera.

    Counting every later unstamped frame as startup would report most of a
    healthy stream as blind, which is exactly backwards: those frames are the
    ordinary gaps the server side interpolates across.
    """
    STARTUP_FRAMES = 5
    buffer = io.BytesIO()
    source = av.open(str(h264_source))
    output = av.open(buffer, mode="w", format="mpegts")

    counter = {"n": 0}

    def intermittent(_packet):
        index = counter["n"]
        counter["n"] += 1
        if index < STARTUP_FRAMES:
            return None, None
        # From the first anchor on, every other packet carries nothing - roughly
        # what a real camera does, where picture timing reaches about half.
        if (index - STARTUP_FRAMES) % 2:
            return None, None
        return prft(BASE_CAPTURE_US + index * FRAME_US), None

    relay = KlvRelay(source, output, side_data=intermittent)
    for index, packet in enumerate(source.demux(relay.in_video_stream)):
        if packet.dts is None:
            continue
        relay.relay(index / FPS, packet)
    output.close()
    source.close()

    assert relay.stats.frames_before_first_anchor == STARTUP_FRAMES
    # The gaps after startup are real gaps, not startup.
    assert relay.stats.klv_packets < relay.stats.video_packets
    assert relay.stats.klv_packets > 0
