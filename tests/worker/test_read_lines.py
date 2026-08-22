import base64
import json
import math
import struct
from unittest.mock import Mock

import aiohttp
import pytest

from eyepop.worker.worker_jobs import _iter_lines


def _stream_reader(limit: int, payload: bytes) -> aiohttp.StreamReader:
    reader = aiohttp.StreamReader(Mock(_reading_paused=False), limit=limit)
    reader.feed_data(payload)
    reader.feed_eof()
    return reader


def _depth_prediction_line(width: int, height: int) -> bytes:
    values = [1.0] * (width * height - 1) + [math.inf]
    depth = {
        "width": width,
        "height": height,
        "values": base64.b64encode(struct.pack(f"<{width * height}f", *values)).decode(),
    }
    return json.dumps({"source_width": width, "source_height": height, "depth": depth}).encode() + b"\n"


# aiohttp's default ClientSession read_bufsize, i.e. what the SDK runs with
_DEFAULT_LIMIT = 65536
# a real 518x518 depth ability emits this map size for a portrait frame
_DEPTH_WIDTH, _DEPTH_HEIGHT = 388, 518


async def test_iter_lines_reads_lines_larger_than_the_read_buffer():
    """A depth map is ~1mb of base64 per frame, far beyond aiohttp's readline() limit."""
    line = _depth_prediction_line(_DEPTH_WIDTH, _DEPTH_HEIGHT)
    assert len(line) > _DEFAULT_LIMIT

    lines = [chunk async for chunk in _iter_lines(_stream_reader(_DEFAULT_LIMIT, line))]

    assert lines == [line]
    prediction = json.loads(lines[0])
    assert prediction["depth"]["width"] == _DEPTH_WIDTH


async def test_readline_would_reject_the_same_payload():
    """Regression guard: readline() is why large predictions failed with 'Chunk too big'."""
    line = _depth_prediction_line(_DEPTH_WIDTH, _DEPTH_HEIGHT)

    with pytest.raises(ValueError):
        await _stream_reader(_DEFAULT_LIMIT, line).readline()


async def test_iter_lines_splits_multiple_lines_across_chunk_boundaries():
    payload = b'{"a": 1}\n{"b": 2}\n{"c": 3}\n'
    reader = aiohttp.StreamReader(Mock(_reading_paused=False), limit=8)
    # feed in fragments that split lines mid-way
    for start in range(0, len(payload), 5):
        reader.feed_data(payload[start:start + 5])
    reader.feed_eof()

    lines = [chunk async for chunk in _iter_lines(reader)]

    assert lines == [b'{"a": 1}\n', b'{"b": 2}\n', b'{"c": 3}\n']


async def test_iter_lines_yields_a_trailing_line_without_newline():
    lines = [chunk async for chunk in _iter_lines(_stream_reader(1024, b'{"a": 1}\n{"b": 2}'))]

    assert lines == [b'{"a": 1}\n', b'{"b": 2}']


async def test_iter_lines_on_empty_body():
    lines = [chunk async for chunk in _iter_lines(_stream_reader(1024, b''))]

    assert lines == []
