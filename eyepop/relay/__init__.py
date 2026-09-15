"""Relay a locally reachable RTSP camera into EyePop with its own capture times.

A camera that cannot be exposed to the internet is read locally and its stream
forwarded to a worker. The container crossing that boundary carries no RTCP, so
the NTP reference the worker would otherwise take from the camera is gone and
predictions come back with no ``captured_at``. This package carries it in-band
instead, as MISB ST 0601 KLV alongside the video.

Nothing here decodes or re-encodes video: packets are copied, and the metadata
rides beside them.
"""

from eyepop.relay.mux import KlvRelay, RelayStats
from eyepop.relay.ntp import CaptureClock
from eyepop.relay.pipe import PipeBuffer
from eyepop.relay.rtsp import (
    INITIAL_BACKOFF_S,
    MAX_BACKOFF_S,
    MAX_PENDING_BYTES,
    MAX_STALL_S,
    READ_TIMEOUT_S,
    BackpressureError,
    CameraError,
    MuxError,
    RelayError,
    RtspRelayStream,
    UploadError,
    rtsp_relay_stream,
)
from eyepop.relay.st0601 import (
    UAS_LDS_UNIVERSAL_LABEL,
    PlatformOrientation,
    SensorPosition,
    encode_st0601,
)

__all__ = [
    "INITIAL_BACKOFF_S",
    "MAX_BACKOFF_S",
    "MAX_PENDING_BYTES",
    "MAX_STALL_S",
    "READ_TIMEOUT_S",
    "UAS_LDS_UNIVERSAL_LABEL",
    "BackpressureError",
    "CameraError",
    "CaptureClock",
    "KlvRelay",
    "MuxError",
    "PipeBuffer",
    "PlatformOrientation",
    "RelayError",
    "RelayStats",
    "RtspRelayStream",
    "SensorPosition",
    "UploadError",
    "rtsp_relay_stream",
    "encode_st0601",
]
