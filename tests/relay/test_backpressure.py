"""What the relay does when the upload cannot keep up (AWSU-260).

The queue between the muxing thread and the upload used to be unbounded, so a
stalled upload buffered MPEG-TS without limit - 7.7 MB per minute of stall on a
1 Mbps camera, per camera. The decision recorded here is to shed load rather
than to buffer it or to block on it:

* The bound is enforced *above* the muxer, by dropping demuxed packets before
  they are relayed. Bounding the queue itself would block the muxer's write,
  and that write happens inside a C callback no stop flag can reach - so the
  shutdown that joins the muxing thread would hang, holding the RTSP socket
  open behind it.
* Whole groups of pictures go, never part of one, so the far side never decodes
  against a reference frame that was never sent.
* An anchor is built from the packet it describes, so a dropped packet takes
  its capture time with it and no anchor is left pointing at a missing frame.
  That is the property AWSU-246 exists to protect.
* Dropping rides out a stall; it does not replace the stream. Past `max_stall_s`
  the session ends so that a caller reconnects, rather than shedding every
  frame forever and looking alive while delivering nothing.
"""

from __future__ import annotations

import struct

import pytest

from eyepop.relay import (
    BackpressureError,
    CameraError,
    KlvRelay,
    PipeBuffer,
    rtsp_relay_stream,
)

#: Small enough that the source overruns it immediately, so the tests exercise
#: a stall rather than wait around for one.
TINY_BOUND = 4096

#: A plausible capture wallclock. The value does not matter; that there is one
#: on every packet does - see `stamped_relay`.
BASE_CAPTURE_US = 1789316993000000


@pytest.fixture
def stamped_relay(monkeypatch):
    """Make the relay behave like a camera that sends a clock reference.

    Only a live RTSP source carries the side data a capture time comes from, so
    a file fixture produces no anchors at all - and with no anchors the KLV
    stream stays empty, which is a different code path from the one production
    runs. These tests want the ordinary one.
    """
    counter = {"n": 0}

    def side_data(_packet):
        index = counter["n"]
        counter["n"] += 1
        return struct.pack("=qi", BASE_CAPTURE_US + index * 40_000, 0), None

    class StampedKlvRelay(KlvRelay):
        def __init__(self, source, output, **kwargs):
            super().__init__(source, output, side_data=side_data, **kwargs)

    monkeypatch.setattr("eyepop.relay.rtsp.KlvRelay", StampedKlvRelay)


@pytest.fixture
def fast_forward(monkeypatch):
    """A clock that jumps a second per reading, so a stall window elapses.

    The alternative is sleeping through `max_stall_s` in a unit test, or setting
    it so small that whether it trips depends on how fast the machine gets
    through a hundred packets. Neither tests the rule.
    """
    state = {"now": 0.0}

    def monotonic() -> float:
        state["now"] += 1.0
        return state["now"]

    monkeypatch.setattr("eyepop.relay.rtsp.time.monotonic", monotonic)


@pytest.fixture
def scripted_backlog(monkeypatch):
    """Drive the relay's view of the backlog from the test, not from timing.

    Whether a stall recovers depends on a reader draining at the right moment,
    which in a test means sleeping and hoping. Scripting the one number the
    relay actually consults makes the state machine deterministic: each entry is
    what the next packet sees, and the last entry repeats forever.
    """

    def script(*values: int) -> None:
        state = {"n": 0}

        def pending(_self) -> int:
            index = min(state["n"], len(values) - 1)
            state["n"] += 1
            return values[index]

        monkeypatch.setattr(PipeBuffer, "pending_bytes", property(pending))

    return script


async def run_with_no_reader(path, **kwargs):
    """Mux the whole source with nothing draining the other end.

    The muxing thread is driven directly rather than through `__aiter__`,
    because what is being simulated is precisely the absence of the reader that
    `__aiter__` provides. Running it inline also keeps the result deterministic.
    """
    stream = await rtsp_relay_stream(str(path), **kwargs)
    try:
        stream._pipe_through()
    finally:
        stream._container.close()
    return stream


def relayed_packets(monkeypatch) -> list[tuple[int, bool]]:
    """Record `(pts, is_keyframe)` for every packet that survived the drop."""
    seen: list[tuple[int, bool]] = []
    original = KlvRelay.relay

    def record(self, elapsed_s, packet):
        seen.append((packet.pts, packet.is_keyframe))
        return original(self, elapsed_s, packet)

    monkeypatch.setattr(KlvRelay, "relay", record)
    return seen


@pytest.mark.asyncio
async def test_a_stalled_upload_no_longer_buffers_without_limit(
    h264_multi_gop_file, stamped_relay
):
    """The defect itself: unbounded growth behind a consumer that never reads."""
    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=3600.0
    )

    assert stream.stats.dropped_packets > 0
    # Generous about the margin and strict about there being one: the bound is
    # checked before relaying rather than during, so it overshoots by whatever
    # the packet that crossed it produced. Unbounded growth would be the whole
    # source, orders of magnitude past this.
    assert stream._pipe.pending_bytes < TINY_BOUND * 16


@pytest.mark.asyncio
async def test_the_backlog_is_visible_even_when_the_camera_sends_no_clock(
    h264_multi_gop_file,
):
    """No `stamped_relay` here: this is the camera with no capture times at all.

    Its KLV stream stays empty, and FFmpeg will hold video for ten seconds
    waiting to interleave against a stream that never produces anything - out
    of reach of the backlog the relay measures, so the bound above would never
    fire and the memory would grow exactly as it did before. Capped explicitly
    when the muxer is opened; this is the test that the cap is still there.
    """
    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=3600.0
    )

    assert stream.stats.klv_packets == 0, "the fixture was not the unstamped case"
    assert stream.stats.dropped_packets > 0, "the backlog was invisible to the relay"


@pytest.mark.asyncio
async def test_nothing_is_dropped_while_the_upload_keeps_up(
    h264_multi_gop_file, stamped_relay
):
    """A relay that sheds load it did not need to shed is worse than the bug.

    The memory was at least recoverable; the frames are not.
    """
    stream = await rtsp_relay_stream(str(h264_multi_gop_file))
    async for _chunk in stream:
        pass

    assert stream.stats.dropped_packets == 0
    assert stream.stats.drop_episodes == 0
    assert stream.stats.video_packets > 0


@pytest.mark.asyncio
async def test_dropping_takes_whole_groups_of_pictures(
    h264_multi_gop_file, stamped_relay, scripted_backlog, monkeypatch
):
    """Never part of one, or the far side decodes against frames never sent.

    Checked on the relayed packets rather than the output bytes: a gap in the
    presentation timestamps is what a dropped group looks like, and the packet
    that reopens the stream after one has to be a keyframe.
    """
    relayed = relayed_packets(monkeypatch)
    # Keeping up, then far behind for a while, then caught up for good.
    scripted_backlog(*([0] * 5 + [TINY_BOUND * 4] * 20 + [0]))

    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=3600.0
    )

    assert stream.stats.dropped_packets > 0, "the stall never happened"
    assert len(relayed) > 30, "the relay never resumed, so there is no gap to check"

    # Derived rather than assumed: the fixture's time base is not one tick per
    # frame, and hard-coding a step would make this pass for the wrong reason.
    step = min(
        pts - previous_pts
        for (previous_pts, _), (pts, _) in zip(relayed, relayed[1:], strict=False)
        if pts > previous_pts
    )

    gaps = 0
    for (previous_pts, _), (pts, is_keyframe) in zip(relayed, relayed[1:], strict=False):
        if pts - previous_pts == step:
            continue
        gaps += 1
        assert is_keyframe, (
            f"relaying resumed at pts {pts} on a non-keyframe: the group "
            f"starting there is missing the reference frame it predicts from"
        )
    assert gaps == 1, f"expected exactly one dropped run, found {gaps}"


@pytest.mark.asyncio
async def test_a_dropped_frame_takes_its_capture_time_with_it(
    h264_multi_gop_file, stamped_relay, scripted_backlog
):
    """The property AWSU-246 exists to protect, under a stall.

    An anchor that outlived its frame would attach a capture time to whichever
    frame landed nearest, which is worse than having none: the value looks right
    and is wrong. Guaranteed structurally - the anchor is built from the packet
    inside `relay()`, so a packet that is never relayed never produces one - and
    asserted here because that guarantee is the reason the drop was put above
    the muxer rather than inside the queue.
    """
    scripted_backlog(*([0] * 5 + [TINY_BOUND * 4] * 20 + [0]))

    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=3600.0
    )

    assert stream.stats.dropped_packets > 0, "the stall never happened"
    # Every packet carries side data in this fixture, so a relayed packet is an
    # anchor and a dropped one is neither. Equality is what says no anchor was
    # left behind by the frame it belonged to.
    assert stream.stats.klv_packets == stream.stats.video_packets


@pytest.mark.asyncio
async def test_the_stall_clock_stops_while_waiting_for_a_keyframe(
    h264_multi_gop_file, stamped_relay, scripted_backlog, fast_forward
):
    """Recovering is not stalling, even though the relay is still dropping.

    Once the upload has caught up the relay keeps dropping until the next
    keyframe, which on a camera with a long keyframe interval is seconds. Timed
    as stall, that wait would end a session that was about to be fine - and on
    a real camera it would do so every time the upload hiccupped.
    """
    # Behind for one packet, caught up from then on. The stall window is far
    # shorter than the wait for the next keyframe that follows.
    scripted_backlog(TINY_BOUND * 4, 0)

    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=5.0
    )

    assert stream.stats.drop_episodes == 1
    assert stream.failure is None or not isinstance(stream.failure, BackpressureError)


@pytest.mark.asyncio
async def test_an_upload_that_never_recovers_ends_the_session(
    h264_multi_gop_file, stamped_relay, fast_forward
):
    """Dropping forever is alive and useless; a new session is how it recovers.

    Without this the relay sheds every frame indefinitely, the caller sees no
    predictions and no error, and the reconnect that would have fixed it never
    fires.
    """
    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=5.0
    )

    assert isinstance(stream.failure, BackpressureError), stream.failure
    # Not a CameraError: the camera is fine, and reporting it as the cause sends
    # whoever is chasing the reconnect loop to the wrong end of the wire.
    assert not isinstance(stream.failure, CameraError)


@pytest.mark.asyncio
async def test_a_stall_that_ends_the_session_still_signals_the_end_of_stream(
    h264_multi_gop_file, stamped_relay, scripted_backlog, fast_forward
):
    """Or the reader blocks forever and the upload never learns it is over.

    The hazard `signal_eof` exists for, on a path that did not exist when it
    was written: leaving the demux loop because the upload fell behind.
    """
    # Scripted, because here a real reader is draining: without it the backlog
    # would dip under the resume mark and the relay would recover, which is the
    # opposite of the case under test.
    scripted_backlog(TINY_BOUND * 4)

    stream = await rtsp_relay_stream(
        str(h264_multi_gop_file), max_pending_bytes=TINY_BOUND, max_stall_s=5.0
    )
    async for _chunk in stream:
        pass

    assert isinstance(stream.failure, BackpressureError), stream.failure


@pytest.mark.asyncio
async def test_a_drop_episode_is_counted_once_however_long_it_lasts(
    h264_multi_gop_file, stamped_relay
):
    """One long stall and fifty brief ones need different fixes."""
    stream = await run_with_no_reader(
        h264_multi_gop_file, max_pending_bytes=TINY_BOUND, max_stall_s=3600.0
    )

    assert stream.stats.drop_episodes == 1
    assert stream.stats.dropped_packets > stream.stats.drop_episodes


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [0, -1])
async def test_an_unusable_pending_bound_is_rejected(h264_file, invalid):
    """A bound of zero or less would shed every packet and relay nothing."""
    with pytest.raises(ValueError, match="max_pending_bytes must be positive"):
        await rtsp_relay_stream(str(h264_file), max_pending_bytes=invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [0, -1.0, float("nan"), float("inf")])
async def test_an_unusable_stall_window_is_rejected(h264_file, invalid):
    """Zero would end the session the first time a chunk was in flight."""
    with pytest.raises(ValueError, match="max_stall_s must be finite and positive"):
        await rtsp_relay_stream(str(h264_file), max_stall_s=invalid)
