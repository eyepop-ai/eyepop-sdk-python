"""Decoding the camera's capture wallclock, against recorded payload shapes.

No RTSP server: the payloads below are the byte layouts FFmpeg produces, built
here so CI needs nothing but the bytes.
"""

import logging
import struct

from eyepop.relay.ntp import (
    NTP_UNIX_OFFSET_SECONDS,
    CaptureClock,
    decode_prft,
    decode_rtcp_sr,
)

CAPTURE_US = 1789316993000000
CAPTURE_SECONDS = CAPTURE_US // 1_000_000


def prft_payload(unix_us: int) -> bytes:
    """``{int64 wallclock_us; int32 flags}``, host byte order."""
    return struct.pack("=qi", unix_us, 0)


def rtcp_sr_payload(unix_seconds: int, fraction: int = 0) -> bytes:
    """FFmpeg reception wallclock, then the 64-bit NTP stamp at offset 8."""
    ntp = ((unix_seconds + NTP_UNIX_OFFSET_SECONDS) << 32) | fraction
    return struct.pack("=q", 0) + struct.pack("=Q", ntp) + bytes(16)


def test_prft_decodes_to_the_wallclock_it_carries():
    assert decode_prft(prft_payload(CAPTURE_US)) == CAPTURE_US


def test_rtcp_sr_reads_the_ntp_stamp_from_offset_eight():
    assert decode_rtcp_sr(rtcp_sr_payload(CAPTURE_SECONDS)) == CAPTURE_US


def test_rtcp_sr_read_big_endian_is_rejected_rather_than_believed():
    """The byte-order trap, caught at runtime and not only in review.

    A big-endian read of this payload decodes to 1992-07-30 - wrong by 34 years,
    but a real-looking date that would sail through a casual eyeball and stamp
    every frame of a stream.
    """
    ntp = (CAPTURE_SECONDS + NTP_UNIX_OFFSET_SECONDS) << 32
    wrong_order = bytes(8) + struct.pack(">Q", ntp) + bytes(16)
    assert decode_rtcp_sr(wrong_order) is None


def test_short_payloads_are_refused_rather_than_guessed_at():
    assert decode_prft(b"\x00" * 4) is None
    assert decode_rtcp_sr(b"\x00" * 8) is None


def test_a_zero_ntp_stamp_is_not_an_anchor():
    assert decode_rtcp_sr(struct.pack("=q", 0) + struct.pack("=Q", 0) + bytes(16)) is None


def test_prft_is_preferred_over_rtcp_sr():
    """PRFT arrives per packet; a sender report arrives once per interval."""
    clock = CaptureClock()
    capture = clock.note(prft_payload(CAPTURE_US), rtcp_sr_payload(CAPTURE_SECONDS + 5), 0.0)
    assert capture is not None
    assert capture.source == "prft"
    assert capture.unix_us == CAPTURE_US


def test_rtcp_sr_is_used_when_prft_is_absent():
    clock = CaptureClock()
    capture = clock.note(None, rtcp_sr_payload(CAPTURE_SECONDS), 0.0)
    assert capture is not None
    assert capture.source == "rtcp_sr"


def test_a_packet_with_no_side_data_reports_no_capture_time():
    """The normal state at the start of every stream, not a failure."""
    clock = CaptureClock()
    assert clock.note(None, None, 0.0) is None
    assert clock.anchor_count == 0


def test_anchors_beginning_partway_through_are_picked_up():
    """The shape of every healthy stream: a blind window, then anchors.

    Named by AWSU-252's acceptance criteria as a fixture this must cover.
    """
    clock = CaptureClock(absence_timeout_s=30.0)
    for frame in range(50):
        assert clock.note(None, None, frame * 0.04) is None
    assert clock.anchor_count == 0

    for frame in range(50, 100):
        capture = clock.note(prft_payload(CAPTURE_US + frame * 40_000), None, frame * 0.04)
        assert capture is not None
    assert clock.anchor_count == 50


def test_no_warning_during_a_normal_startup_window(caplog):
    """A warning that fires on every healthy stream hides the case it reports.

    The direct RTSP path has the same blind window, measured at 3.3-8.6s on
    AWSU-248, so anything inside that must stay silent.
    """
    clock = CaptureClock(absence_timeout_s=30.0)
    with caplog.at_level(logging.WARNING):
        for frame in range(250):  # 10s at 25fps
            clock.note(None, None, frame * 0.04)
    assert caplog.records == []


def test_a_camera_that_never_sends_sender_reports_warns_once(caplog):
    clock = CaptureClock(absence_timeout_s=1.0)
    with caplog.at_level(logging.WARNING):
        for frame in range(200):
            clock.note(None, None, frame * 0.04)
    assert len(caplog.records) == 1
    assert "no RTCP sender reports" in caplog.records[0].message


def test_the_absence_warning_names_the_camera_as_the_cause(caplog):
    """Otherwise the user is left guessing why captured_at is missing."""
    clock = CaptureClock(absence_timeout_s=0.5)
    with caplog.at_level(logging.WARNING):
        for frame in range(50):
            clock.note(None, None, frame * 0.04)
    message = caplog.records[0].message
    assert "captured_at" in message
    assert "not a relay failure" in message


def test_an_anchor_arriving_late_prevents_the_absence_warning(caplog):
    clock = CaptureClock(absence_timeout_s=1.0)
    with caplog.at_level(logging.WARNING):
        for frame in range(20):
            clock.note(None, None, frame * 0.04)
        clock.note(prft_payload(CAPTURE_US), None, 0.8)
        for frame in range(100):
            clock.note(None, None, 1.0 + frame * 0.04)
    assert caplog.records == []


def test_undecodable_side_data_warns_differently_from_absent_side_data(caplog):
    """Side data that arrives and will not decode is this side's problem.

    Reporting it as "the camera sent nothing" would send anyone debugging it to
    the wrong end of the system.
    """
    clock = CaptureClock()
    with caplog.at_level(logging.WARNING):
        clock.note(prft_payload(1), None, 0.0)
    assert len(caplog.records) == 1
    assert "implausible" in caplog.records[0].message
    assert "RTCP sender reports" not in caplog.records[0].message
