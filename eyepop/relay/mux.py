"""Mux ST 0601 KLV alongside video that is copied, never re-encoded.

The video packets are passed straight through; only the container changes. The
KLV rides beside them on its own stream, one local set per frame, which is what
lets the worker match an anchor to a frame exactly instead of interpolating
between sparse ones.

Requires PyAV (the ``relay`` extra).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import av
from av.container import InputContainer, OutputContainer
from av.packet import Packet
from av.video.stream import VideoStream

from eyepop.relay.ntp import CaptureClock, CaptureTime
from eyepop.relay.st0601 import PlatformOrientation, SensorPosition, encode_st0601

logger = logging.getLogger(__name__)

__all__ = ["KlvRelay", "RelayStats", "extract_side_data"]


def extract_side_data(packet: Packet) -> tuple[bytes | None, bytes | None]:
    """The two side data payloads a capture time can come from, if present."""
    prft = bytes(packet.get_sidedata("prft")) if packet.has_sidedata("prft") else None
    rtcp = bytes(packet.get_sidedata("rtcp_sr")) if packet.has_sidedata("rtcp_sr") else None
    return prft, rtcp


@dataclass
class RelayStats:
    """What a relay run did, for logging and for tests to assert on."""

    video_packets: int = 0
    klv_packets: int = 0
    #: Frames relayed before the first capture time was available. Expected to be
    #: non-zero on a healthy camera: the relay does not wait.
    frames_before_first_anchor: int = 0


class KlvRelay:
    """Copies video into an MPEG-TS output and writes ST 0601 beside it."""

    def __init__(
        self,
        source: InputContainer,
        output: OutputContainer,
        clock: CaptureClock | None = None,
        platform: PlatformOrientation | None = None,
        sensor: SensorPosition | None = None,
        side_data: Callable[[Packet], tuple[bytes | None, bytes | None]] = extract_side_data,
    ) -> None:
        self._source = source
        self._output = output
        self._clock = clock if clock is not None else CaptureClock()
        self._platform = platform
        self._sensor = sensor
        # A seam, so the muxing can be exercised against a file. Only a live
        # RTSP source carries these payloads, and a test that needs one is a
        # test that does not run.
        self._side_data = side_data

        #: Public: callers demux from this to drive relay().
        self.in_video_stream: VideoStream = source.streams.video[0]
        self._out_video = output.add_stream_from_template(template=self.in_video_stream)
        self._out_klv = output.add_data_stream(codec_name="klv")
        # A data stream's time base is None at creation. Mirroring the video
        # stream's is what makes the two sets of timestamps directly comparable,
        # and comparing them is the whole basis of matching an anchor to a frame
        # on the far side. A source that reports no time base at all cannot be
        # relayed with capture times, so say so here rather than emit anchors
        # nothing can line up.
        time_base = self.in_video_stream.time_base
        if time_base is None:
            raise ValueError("source video stream has no time base; cannot align KLV anchors to it")
        self._out_klv.time_base = time_base
        self._time_base = time_base

        self.stats = RelayStats()

    def _klv_packet(self, video_packet: Packet, capture: CaptureTime) -> Packet:
        payload = encode_st0601(capture.unix_us, self._platform, self._sensor)
        packet = av.Packet(payload)
        packet.stream = self._out_klv
        packet.time_base = self._time_base
        # PTS carries the meaning - which frame this anchor describes - while DTS
        # only has to be monotonic. Taking DTS from the video packet's PTS instead
        # fails once B-frames make PTS non-monotonic, and it fails partway into
        # the stream rather than on the first packet, so a short run will not
        # catch it.
        packet.pts = video_packet.pts
        packet.dts = video_packet.dts
        return packet

    def relay(self, elapsed_s: float, packet: Packet) -> None:
        """Copy one demuxed video packet through, with its anchor beside it."""
        prft, rtcp = self._side_data(packet)
        capture = self._clock.note(prft, rtcp, elapsed_s)

        packet.stream = self._out_video
        self._output.mux(packet)
        self.stats.video_packets += 1

        if capture is None:
            # Uploading immediately and letting the leading frames go out
            # unstamped is deliberate, and it matches what the direct RTSP path
            # does: it has the same blind window while it waits for its first
            # sender report.
            #
            # Only while that window is still open. Picture timing arrives on
            # roughly half of packets, so counting every later unstamped frame
            # here would report most of a healthy stream as startup.
            if self._clock.anchor_count == 0:
                self.stats.frames_before_first_anchor += 1
            return

        # Never before the video packet above. A KLV packet muxed first leaves
        # the muxer with no established stream to interleave against and the
        # write fails outright.
        self._output.mux(self._klv_packet(packet, capture))
        self.stats.klv_packets += 1
