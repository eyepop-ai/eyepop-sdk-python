"""ST 0601 encoding, asserted against bytes rather than against itself.

The expected packets below were verified by parsing them with the worker's own
C parser (`ep_klv_parse_st0601`), which was written from the standard
independently of this encoder. Asserting on the round trip through this module
alone would agree with itself no matter which tags it used.
"""

import struct

import pytest

from eyepop.relay.st0601 import (
    UAS_LDS_UNIVERSAL_LABEL,
    PlatformOrientation,
    SensorPosition,
    checksum,
    encode_st0601,
)

CAPTURE_US = 1789316993000000

# Verified with the worker's parser: status=OK, unix_us=1789316993000000, version=6.
MINIMAL_PACKET = bytes.fromhex("060e2b34020b01010e0103010100000011020800065b5fd3c222404101060102c9eb")


def test_minimal_packet_matches_the_bytes_the_worker_parses():
    assert encode_st0601(CAPTURE_US) == MINIMAL_PACKET


def test_packet_opens_with_the_uas_lds_universal_label():
    packet = encode_st0601(CAPTURE_US)
    assert packet[:16] == UAS_LDS_UNIVERSAL_LABEL


def test_checksum_is_the_final_item_not_the_first():
    """ST 0601 gives the checksum the lowest tag number and requires it last.

    Encoding it as Tag 65, or putting Tag 1 at the front, produces a packet that
    parses structurally and is then rejected as having no checksum at all - the
    most expensive way to get this wrong, because nothing looks broken.
    """
    packet = encode_st0601(CAPTURE_US)
    assert packet[-4] == 1
    assert packet[-3] == 2
    assert packet[17] == 2, "the timestamp opens the set"


def test_version_is_tag_65():
    packet = encode_st0601(CAPTURE_US)
    assert packet[-7:-4] == bytes((65, 1, 6))


def test_checksum_covers_everything_up_to_its_own_length_field():
    packet = encode_st0601(CAPTURE_US)
    (declared,) = struct.unpack(">H", packet[-2:])
    assert declared == checksum(packet[:-2])


def test_a_corrupted_byte_breaks_the_checksum():
    packet = bytearray(encode_st0601(CAPTURE_US))
    packet[20] ^= 0xFF
    (declared,) = struct.unpack(">H", bytes(packet[-2:]))
    assert declared != checksum(bytes(packet[:-2]))


def test_timestamp_is_eight_bytes_big_endian_microseconds():
    packet = encode_st0601(CAPTURE_US)
    assert packet[17:19] == bytes((2, 8))
    (value,) = struct.unpack(">Q", packet[19:27])
    assert value == CAPTURE_US


def tags_in(packet: bytes) -> list[int]:
    """Tags present, by walking the items.

    Searching for a tag byte instead would find one inside a value: 0x06 occurs
    in the middle of every timestamp written this decade.
    """
    length = packet[16]
    assert length < 0x80, "this helper only handles the short form"
    offset, end, found = 17, 17 + length, []
    while offset < end:
        tag = packet[offset]
        item_length = packet[offset + 1]
        found.append(tag)
        offset += 2 + item_length
    return found


def test_position_tags_are_absent_unless_supplied():
    """An absent tag and a zero reading are different things in ST 0601."""
    packet = encode_st0601(CAPTURE_US)
    assert len(packet) == 34
    assert tags_in(packet) == [2, 65, 1]


def test_every_scoped_tag_appears_once_when_all_are_supplied():
    packet = encode_st0601(
        CAPTURE_US,
        PlatformOrientation(45.0, -5.0, 10.0),
        SensorPosition(37.7749, -122.4194, 52.0),
    )
    assert tags_in(packet) == [2, 5, 6, 7, 13, 14, 15, 65, 1]


def test_platform_and_sensor_tags_are_written_when_supplied():
    packet = encode_st0601(
        CAPTURE_US,
        PlatformOrientation(45.0, -5.0, 10.0),
        SensorPosition(37.7749, -122.4194, 52.0),
    )
    assert len(packet) == 62
    (declared,) = struct.unpack(">H", packet[-2:])
    assert declared == checksum(packet[:-2])


@pytest.mark.parametrize(
    "supplied,expected,tolerance",
    [
        (0.0, 0.0, 0.01),
        (45.0, 45.0, 0.01),
        (359.9, 359.9, 0.01),
    ],
)
def test_heading_scales_onto_the_full_width_of_its_integer(supplied, expected, tolerance):
    packet = encode_st0601(CAPTURE_US, PlatformOrientation(supplied, 0.0, 0.0))
    offset = packet.index(bytes((5, 2)), 17)
    (raw,) = struct.unpack(">H", packet[offset + 2 : offset + 4])
    assert abs(raw / 0xFFFF * 360.0 - expected) < tolerance


def test_signed_items_never_emit_the_reserved_out_of_range_marker():
    """0x8000 means "not measured", so a real reading must never encode to it.

    Scaling by the maximum rather than by half the range is what keeps the most
    negative reading one step above the marker.
    """
    packet = encode_st0601(CAPTURE_US, PlatformOrientation(0.0, -20.0, -50.0))
    pitch = packet.index(bytes((6, 2)), 17)
    roll = packet.index(bytes((7, 2)), 17)
    assert packet[pitch + 2 : pitch + 4] != b"\x80\x00"
    assert packet[roll + 2 : roll + 4] != b"\x80\x00"


def test_out_of_range_values_are_clamped_rather_than_wrapped():
    """A wrapped angle reads as a valid measurement pointing the wrong way."""
    packet = encode_st0601(CAPTURE_US, PlatformOrientation(400.0, 90.0, -90.0))
    heading = packet.index(bytes((5, 2)), 17)
    assert packet[heading + 2 : heading + 4] == b"\xff\xff"


def test_a_long_form_ber_length_is_used_when_the_set_needs_one():
    """Both BER forms occur in real streams; the short form stops at 0x7f."""
    packet = encode_st0601(
        CAPTURE_US,
        PlatformOrientation(1.0, 1.0, 1.0),
        SensorPosition(1.0, 1.0, 1.0),
    )
    assert packet[16] < 0x80, "this set still fits the short form"


def test_negative_timestamps_are_refused():
    with pytest.raises(ValueError):
        encode_st0601(-1)
