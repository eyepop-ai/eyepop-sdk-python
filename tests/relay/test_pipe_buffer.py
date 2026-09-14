"""The hand-off between the muxing thread and the uploading thread."""

from __future__ import annotations

import threading

from eyepop.relay.pipe import PipeBuffer


def test_reads_what_was_written():
    pipe = PipeBuffer()
    pipe.write(b"hello")
    assert pipe.read(5) == b"hello"


def test_read_returns_empty_once_the_writer_is_done():
    """Without this the reader blocks forever and the upload never ends.

    A finite source reaching its end, a camera disconnecting, or anything raised
    in the muxing thread all arrive here as the same thing: no more writes.
    """
    pipe = PipeBuffer()
    pipe.write(b"payload")
    pipe.signal_eof()

    assert pipe.read(7) == b"payload"
    assert pipe.read(1) == b""
    assert pipe.read(1) == b"", "end of stream is not a one-shot"


def test_queued_data_survives_the_end_of_stream():
    """Signalling the end must not discard what is still in flight."""
    pipe = PipeBuffer()
    pipe.write(b"one")
    pipe.write(b"two")
    pipe.signal_eof()

    assert pipe.read(3) == b"one"
    assert pipe.read(3) == b"two"
    assert pipe.read(3) == b""


def test_a_blocked_reader_is_released_by_the_writer_finishing():
    """The shape of the hang: the reader is already waiting when the source ends."""
    pipe = PipeBuffer()
    result: list[bytes] = []

    def read_one():
        result.append(pipe.read(16))

    reader = threading.Thread(target=read_one)
    reader.start()
    pipe.signal_eof()
    reader.join(timeout=5)

    assert not reader.is_alive(), "reader never woke: this is the indefinite upload hang"
    assert result == [b""]


def test_a_zero_length_read_returns_at_once_instead_of_waiting():
    """The io contract, and a deadlock if it is not honoured.

    A caller asking for nothing is not asking to be blocked until somebody
    writes something.
    """
    pipe = PipeBuffer()
    finished: list[bytes] = []

    reader = threading.Thread(target=lambda: finished.append(pipe.read(0)))
    reader.start()
    reader.join(timeout=5)

    assert not reader.is_alive(), "read(0) blocked waiting for data it did not ask for"
    assert finished == [b""]


def test_reading_everything_crosses_chunk_boundaries():
    """`read(-1)` means to the end of the stream, not to the end of one write.

    Stopping at the first chunk truncates silently for any caller using the
    default size, which is every caller that does not happen to pass one.
    """
    pipe = PipeBuffer()
    pipe.write(b"one")
    pipe.write(b"two")
    pipe.write(b"three")
    pipe.signal_eof()

    assert pipe.read(-1) == b"onetwothree"
    assert pipe.read(-1) == b""


def test_readall_matches_reading_everything():
    pipe = PipeBuffer()
    pipe.write(b"alpha")
    pipe.write(b"beta")
    pipe.signal_eof()

    assert pipe.readall() == b"alphabeta"


def test_a_sized_read_never_returns_more_than_asked_for():
    pipe = PipeBuffer()
    pipe.write(b"abcdef")
    pipe.signal_eof()

    assert pipe.read(3) == b"abc"
    assert pipe.read(3) == b"def"
    assert pipe.read(3) == b""


def test_a_sized_read_is_satisfied_from_one_chunk_at_a_time():
    """Short reads are allowed; what matters is that nothing is dropped."""
    pipe = PipeBuffer()
    pipe.write(b"ab")
    pipe.write(b"cd")
    pipe.signal_eof()

    collected = b""
    while chunk := pipe.read(16):
        collected += chunk
    assert collected == b"abcd"


def test_the_buffer_reports_itself_readable():
    assert PipeBuffer().readable()


def test_an_empty_write_does_not_end_the_stream():
    """A zero-length write is legitimate; a muxer flushing nothing produces one.

    Treating it as end-of-stream discards everything written afterwards, and
    does so silently - the read simply returns short and the caller has no way
    to tell that from a stream that really ended.
    """
    pipe = PipeBuffer()
    pipe.write(b"before")
    pipe.write(b"")
    pipe.write(b"after")
    pipe.signal_eof()

    assert pipe.readall() == b"beforeafter"


def test_an_empty_write_before_any_data_is_skipped():
    pipe = PipeBuffer()
    pipe.write(b"")
    pipe.write(b"data")
    pipe.signal_eof()

    assert pipe.read(-1) == b"data"


def test_consecutive_empty_writes_are_skipped():
    pipe = PipeBuffer()
    pipe.write(b"")
    pipe.write(b"")
    pipe.write(b"payload")
    pipe.signal_eof()

    assert pipe.read(16) == b"payload"
