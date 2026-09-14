"""Encode MISB ST 0601 UAS Datalink Local Sets.

Deliberately free of any dependency on PyAV or on a container: this is bytes in,
bytes out, so it is testable without an RTSP source and reusable by anything that
needs to write ST 0601.

The scaling below is the inverse of what the worker's parser applies, and the
two have to agree exactly or every reading comes back subtly wrong while looking
entirely plausible. Where a value is mapped onto an integer's full width, the
divisor is the maximum rather than half the range, because ST 0601 reserves the
negative extreme of a signed item as an out-of-range marker instead of a
reading.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

__all__ = [
    "UAS_LDS_UNIVERSAL_LABEL",
    "UAS_LDS_VERSION",
    "PlatformOrientation",
    "SensorPosition",
    "checksum",
    "encode_st0601",
]

#: SMPTE 336M universal label for the ST 0601 UAS Datalink Local Set.
UAS_LDS_UNIVERSAL_LABEL = bytes.fromhex("060e2b34020b01010e0103010100 0000".replace(" ", ""))

#: Value written into Tag 65. ST 0601 revision the encoding below conforms to.
UAS_LDS_VERSION = 6

# ST 0601 gives the checksum the lowest tag number and simultaneously requires it
# to be the final item, so tag order and packet order disagree. Encoding these
# the other way round - version as Tag 1, checksum as Tag 65 - produces packets
# that parse structurally and are then rejected as having no checksum at all.
_TAG_CHECKSUM = 1
_TAG_UNIX_TIMESTAMP = 2
_TAG_PLATFORM_HEADING = 5
_TAG_PLATFORM_PITCH = 6
_TAG_PLATFORM_ROLL = 7
_TAG_SENSOR_LATITUDE = 13
_TAG_SENSOR_LONGITUDE = 14
_TAG_SENSOR_TRUE_ALTITUDE = 15
_TAG_UAS_LDS_VERSION = 65


@dataclass(frozen=True)
class PlatformOrientation:
    """Tags 5, 6 and 7. Degrees."""

    heading_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class SensorPosition:
    """Tags 13, 14 and 15. Degrees and metres."""

    latitude_deg: float
    longitude_deg: float
    true_altitude_m: float


def _ber_length(n: int) -> bytes:
    """BER length: short form below 0x80, long form above it."""
    if n < 0x80:
        return bytes((n,))
    encoded = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes((0x80 | len(encoded),)) + encoded


def _item(tag: int, value: bytes) -> bytes:
    return bytes((tag,)) + _ber_length(len(value)) + value


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _scale_unsigned(value: float, length: int, low: float, high: float) -> bytes:
    full = (1 << (length * 8)) - 1
    ratio = (_clamp(value, low, high) - low) / (high - low)
    return int(round(ratio * full)).to_bytes(length, "big")


def _scale_signed(value: float, length: int, extent: float) -> bytes:
    limit = (1 << (length * 8 - 1)) - 1
    raw = int(round(_clamp(value, -extent, extent) / extent * limit))
    return raw.to_bytes(length, "big", signed=True)


def checksum(packet_through_length: bytes) -> int:
    """ST 0601 checksum: a running 16-bit sum with alternating byte placement.

    Covers the packet from the first byte of the universal label through the
    checksum item's own length field, so the caller passes everything written so
    far and appends the result.
    """
    total = 0
    for index, byte in enumerate(packet_through_length):
        total = (total + (byte << (8 * ((index + 1) % 2)))) & 0xFFFF
    return total


def encode_st0601(
    unix_timestamp_us: int,
    platform: PlatformOrientation | None = None,
    sensor: SensorPosition | None = None,
) -> bytes:
    """One ST 0601 local set carrying a capture time and, optionally, position.

    Position items are written only when supplied. An absent tag and a zero
    reading are different things in ST 0601, and fabricating a zero would be
    reported downstream as a measurement.
    """
    if unix_timestamp_us < 0:
        raise ValueError("unix_timestamp_us must not be negative")

    # Tag 2 opens the set and Tag 1 closes it, which is the standard's order
    # rather than a preference.
    items = _item(_TAG_UNIX_TIMESTAMP, struct.pack(">Q", unix_timestamp_us))

    if platform is not None:
        items += _item(_TAG_PLATFORM_HEADING, _scale_unsigned(platform.heading_deg, 2, 0.0, 360.0))
        items += _item(_TAG_PLATFORM_PITCH, _scale_signed(platform.pitch_deg, 2, 20.0))
        items += _item(_TAG_PLATFORM_ROLL, _scale_signed(platform.roll_deg, 2, 50.0))

    if sensor is not None:
        items += _item(_TAG_SENSOR_LATITUDE, _scale_signed(sensor.latitude_deg, 4, 90.0))
        items += _item(_TAG_SENSOR_LONGITUDE, _scale_signed(sensor.longitude_deg, 4, 180.0))
        items += _item(
            _TAG_SENSOR_TRUE_ALTITUDE, _scale_unsigned(sensor.true_altitude_m, 2, -900.0, 19000.0)
        )

    items += _item(_TAG_UAS_LDS_VERSION, bytes((UAS_LDS_VERSION,)))

    # The checksum item is part of the length it is covered by, so the declared
    # length has to include the four bytes the item itself occupies.
    total_value_length = len(items) + 4
    through_length = (
        UAS_LDS_UNIVERSAL_LABEL + _ber_length(total_value_length) + items + bytes((_TAG_CHECKSUM, 2))
    )
    return through_length + struct.pack(">H", checksum(through_length))
