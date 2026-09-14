"""The hand-off between the thread that muxes and the thread that uploads.

Kept out of the example because the end-of-stream behaviour is the part worth
getting right: the reader blocks indefinitely by design, so the writer has to
say when it is done or the upload never observes the end of the stream.
"""

from __future__ import annotations

import io
import queue

__all__ = ["PipeBuffer"]


class PipeBuffer(io.RawIOBase):
    """A blocking queue written by one thread and read by another."""

    _EOF = object()

    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.buffer: bytes = b""
        self._at_eof = False

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        if isinstance(b, str):
            b = b.encode("utf-8")
        self.queue.put(b)
        return len(b)

    def signal_eof(self) -> None:
        """No more writes are coming.

        Not an override of ``close()``: that marks the object closed, and the
        reader is a different thread still draining what is already queued.
        """
        self.queue.put(self._EOF)

    def read(self, n: int = -1) -> bytes:
        if not self.buffer:
            if self._at_eof:
                return b""
            # Blocks until data is available.
            chunk = self.queue.get(block=True, timeout=None)
            if chunk is self._EOF:
                self._at_eof = True
                return b""
            self.buffer = chunk

        if n < 0:
            res, self.buffer = self.buffer, b""
            return res

        res = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return res
