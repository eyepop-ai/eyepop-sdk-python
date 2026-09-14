"""A minimal MPEG-TS reader, so the relay's output is checked against bytes.

FFmpeg's own tooling misreports this payload: its mpegts demuxer skips a 5-byte
SMPTE RP 217 metadata AU cell header that its muxer never writes, so ffprobe
reports a 34-byte ST 0601 set as 29 bytes and ``ffmpeg -f data`` drops the first
five bytes of the universal label. Verifying with ffprobe would mean chasing a
corruption bug that is not there.

This reassembles PES payloads straight out of the transport stream instead, which
is what GStreamer sees.
"""

from __future__ import annotations

TS_PACKET_SIZE = 188
SYNC_BYTE = 0x47


def _pes_payloads(data: bytes) -> dict[int, list[bytes]]:
    """PES payloads per PID, in order."""
    pending: dict[int, bytearray] = {}
    done: dict[int, list[bytes]] = {}

    for start in range(0, len(data) - TS_PACKET_SIZE + 1, TS_PACKET_SIZE):
        packet = data[start : start + TS_PACKET_SIZE]
        if packet[0] != SYNC_BYTE:
            raise ValueError(f"lost transport stream sync at byte {start}")

        payload_unit_start = bool(packet[1] & 0x40)
        pid = ((packet[1] & 0x1F) << 8) | packet[2]
        adaptation = (packet[3] >> 4) & 0x03
        if adaptation in (0, 2):  # no payload
            continue

        offset = 4
        if adaptation == 3:
            offset += 1 + packet[4]
        if offset >= TS_PACKET_SIZE:
            continue
        body = packet[offset:]

        if payload_unit_start:
            if pid in pending:
                done.setdefault(pid, []).append(bytes(pending.pop(pid)))
            pending[pid] = bytearray(body)
        elif pid in pending:
            pending[pid] += body

    for pid, buffered in pending.items():
        done.setdefault(pid, []).append(bytes(buffered))
    return done


def _strip_pes_header(pes: bytes) -> tuple[int, bytes] | None:
    """``(stream_id, payload)`` for one PES packet, or None if it is not one."""
    if len(pes) < 9 or pes[0:3] != b"\x00\x00\x01":
        return None
    stream_id = pes[3]
    header_length = pes[8]
    return stream_id, pes[9 + header_length :]


def klv_payloads(data: bytes, universal_label: bytes) -> list[bytes]:
    """Every KLV set in the stream, byte for byte as it was written.

    Found by the universal label rather than by reading the PMT: it needs no
    table parsing and it fails loudly if the label is wrong, which is the thing
    most worth catching.
    """
    found: list[bytes] = []
    for payloads in _pes_payloads(data).values():
        for pes in payloads:
            parsed = _strip_pes_header(pes)
            if parsed is None:
                continue
            stream_id, payload = parsed
            if payload.startswith(universal_label):
                found.append(payload)
    return found


def stream_ids(data: bytes) -> set[int]:
    """PES stream_ids present. Private stream 1 (0xBD) and metadata (0xFC)."""
    ids: set[int] = set()
    for payloads in _pes_payloads(data).values():
        for pes in payloads:
            parsed = _strip_pes_header(pes)
            if parsed is not None:
                ids.add(parsed[0])
    return ids
