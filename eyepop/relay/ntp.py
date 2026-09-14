"""Read the camera's capture wallclock out of RTSP packet side data.

FFmpeg attaches two things worth reading. ``prft`` carries a wallclock in
microseconds and arrives on most packets, because FFmpeg already interpolates
the RTCP anchor internally and republishes it per packet - so there is nothing
to interpolate here. ``rtcp_sr`` carries the raw 64-bit NTP timestamp and
arrives once per sender-report interval; it is the fallback and the cross-check.

Nothing in this module imports PyAV, so it is testable against recorded
payloads without an RTSP server.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "CaptureTime",
    "CaptureClock",
    "decode_prft",
    "decode_rtcp_sr",
]

#: Seconds between 1900-01-01 and 1970-01-01.
NTP_UNIX_OFFSET_SECONDS = 2_208_988_800

# Both payloads are written in HOST byte order, not network order. Reading
# rtcp_sr big-endian yields 1992-07-30 for a 2026 capture - wrong, but plausible
# enough to survive a casual look, which is why _is_plausible exists below.
_HOST = "="

# A decoded capture time outside this range is a decoding error rather than a
# camera with a bad clock: the byte-order mistake above lands ~34 years off, and
# every other way of misreading these payloads lands further out still.
_PLAUSIBLE_FROM_US = 1_577_836_800_000_000  # 2020-01-01
_PLAUSIBLE_UNTIL_US = 4_102_444_800_000_000  # 2100-01-01

# How long to wait before declaring a camera to have no usable NTP at all.
#
# "No anchor yet" and "this camera never sends one" look identical at the start
# of every stream, and the direct-RTSP path has the same blind window - AWSU-248
# measured 3.3-8.6s there, varying with where the client joins the sender-report
# cycle, on a synthetic rig that is a lower bound on a real camera. Sender
# reports arrived about every 5-7s in that measurement, so 30s is four to six
# opportunities missed rather than one unlucky join. Warning any sooner would
# fire on healthy streams, and a warning that fires every time hides the case it
# exists to report.
DEFAULT_ABSENCE_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class CaptureTime:
    """When the camera says a packet was captured."""

    unix_us: int
    #: Which side data it came from, for diagnostics and for the cross-check.
    source: str


def _implausible(unix_us: int) -> bool:
    return not (_PLAUSIBLE_FROM_US <= unix_us < _PLAUSIBLE_UNTIL_US)


def decode_prft(payload: bytes) -> int | None:
    """Unix microseconds from an ``AV_PKT_DATA_PRFT`` payload.

    Layout is ``{int64 wallclock_us; int32 flags}``. Only the wallclock is read;
    a shorter payload is a different FFmpeg build's shape and is refused rather
    than guessed at.
    """
    if len(payload) < 8:
        return None
    (wallclock_us,) = struct.unpack_from(f"{_HOST}q", payload, 0)
    if wallclock_us <= 0 or _implausible(wallclock_us):
        return None
    return wallclock_us


def decode_rtcp_sr(payload: bytes) -> int | None:
    """Unix microseconds from an ``AV_PKT_DATA_RTCP_SR`` payload.

    The 64-bit NTP timestamp sits at offset 8, not at the start: the first eight
    bytes are FFmpeg's own reception wallclock.
    """
    if len(payload) < 16:
        return None
    (ntp,) = struct.unpack_from(f"{_HOST}Q", payload, 8)
    if ntp == 0:
        return None
    seconds = (ntp >> 32) - NTP_UNIX_OFFSET_SECONDS
    fraction = (ntp & 0xFFFFFFFF) / 2**32
    if seconds < 0:
        return None
    unix_us = int(round((seconds + fraction) * 1_000_000))
    if _implausible(unix_us):
        return None
    return unix_us


class CaptureClock:
    """Turns per-packet side data into capture times, or into an honest nothing.

    Returning ``None`` is a normal state, not a failure: it is what every stream
    reports until its first anchor arrives, and the relay is expected to keep
    uploading through that window rather than hold a live camera hostage to an
    anchor that may never come.
    """

    def __init__(self, absence_timeout_s: float = DEFAULT_ABSENCE_TIMEOUT_S) -> None:
        self._absence_timeout_s = absence_timeout_s
        self._first_seen_at: float | None = None
        self._anchors = 0
        self._implausible_payloads = 0
        self._warned_absent = False
        self._warned_implausible = False

    @property
    def anchor_count(self) -> int:
        return self._anchors

    def note(
        self,
        prft: bytes | None,
        rtcp_sr: bytes | None,
        elapsed_s: float,
    ) -> CaptureTime | None:
        """Capture time for one packet, given whichever side data it carried.

        ``elapsed_s`` is seconds since the relay opened the source, on any
        monotonic clock. It only drives the absence warning.
        """
        if self._first_seen_at is None:
            self._first_seen_at = elapsed_s

        capture = self._decode(prft, rtcp_sr)
        if capture is not None:
            self._anchors += 1
            return capture

        self._warn_if_absent(elapsed_s)
        return None

    def _decode(self, prft: bytes | None, rtcp_sr: bytes | None) -> CaptureTime | None:
        # PRFT first: it arrives on most packets, where a sender report arrives
        # once per interval.
        if prft is not None:
            unix_us = decode_prft(prft)
            if unix_us is not None:
                return CaptureTime(unix_us, "prft")
            self._note_implausible(len(prft), "prft")

        if rtcp_sr is not None:
            unix_us = decode_rtcp_sr(rtcp_sr)
            if unix_us is not None:
                return CaptureTime(unix_us, "rtcp_sr")
            self._note_implausible(len(rtcp_sr), "rtcp_sr")

        return None

    def _note_implausible(self, size: int, source: str) -> None:
        self._implausible_payloads += 1
        if self._warned_implausible:
            return
        self._warned_implausible = True
        # Distinct from the absence warning on purpose. Side data that arrives
        # and cannot be decoded is a decoding problem on this side; side data
        # that never arrives is the camera's. Reporting both as "no timestamps"
        # would send anyone debugging it to the wrong end.
        logger.warning(
            "Discarding %s side data that decodes to an implausible capture time "
            "(%d bytes). The stream will be relayed without capture times.",
            source,
            size,
        )

    def _warn_if_absent(self, elapsed_s: float) -> None:
        if self._warned_absent or self._anchors > 0:
            return
        if self._first_seen_at is None or elapsed_s - self._first_seen_at < self._absence_timeout_s:
            return
        self._warned_absent = True
        logger.warning(
            "No capture time from this camera after %.0fs: it sent no RTCP sender reports. "
            "The stream is being relayed without capture times, so predictions will have no "
            "captured_at. This is the camera's NTP configuration, not a relay failure.",
            self._absence_timeout_s,
        )
