"""The hand-off between the muxing thread and the uploading thread.

Imported from the example by path: the relay's reference implementation lives
there until AWSU-255 moves it, and the end-of-stream behaviour is worth holding
still in the meantime.
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "relay_example.py"


@pytest.fixture(scope="module")
def pipe_buffer_cls():
    spec = importlib.util.spec_from_file_location("relay_example_for_test", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PipeBuffer


def test_reads_what_was_written(pipe_buffer_cls):
    pipe = pipe_buffer_cls()
    pipe.write(b"hello")
    assert pipe.read(5) == b"hello"


def test_read_returns_empty_once_the_writer_is_done(pipe_buffer_cls):
    """Without this the reader blocks forever and the upload never ends.

    A finite source reaching its end, a camera disconnecting, or anything raised
    in the muxing thread all arrive here as the same thing: no more writes.
    """
    pipe = pipe_buffer_cls()
    pipe.write(b"payload")
    pipe.signal_eof()

    assert pipe.read(7) == b"payload"
    assert pipe.read(1) == b""
    assert pipe.read(1) == b"", "end of stream is not a one-shot"


def test_queued_data_survives_the_end_of_stream(pipe_buffer_cls):
    """Signalling the end must not discard what is still in flight."""
    pipe = pipe_buffer_cls()
    pipe.write(b"one")
    pipe.write(b"two")
    pipe.signal_eof()

    assert pipe.read(3) == b"one"
    assert pipe.read(3) == b"two"
    assert pipe.read(3) == b""


def test_a_blocked_reader_is_released_by_the_writer_finishing(pipe_buffer_cls):
    """The shape of the hang: the reader is already waiting when the source ends."""
    pipe = pipe_buffer_cls()
    result: list[bytes] = []

    def read_one():
        result.append(pipe.read(16))

    reader = threading.Thread(target=read_one)
    reader.start()
    pipe.signal_eof()
    reader.join(timeout=5)

    assert not reader.is_alive(), "reader never woke: this is the indefinite upload hang"
    assert result == [b""]
