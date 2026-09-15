"""The hand-off between the thread that muxes and the thread that uploads.

Kept out of the example because the end-of-stream behaviour is the part worth
getting right: the reader blocks indefinitely by design, so the writer has to
say when it is done or the upload never observes the end of the stream.

Writes never block, deliberately. An upload that stalls is shed upstream, by
the relay loop dropping whole groups of pictures before they are ever muxed -
see :attr:`PipeBuffer.pending_bytes` and ``eyepop.relay.rtsp``. Bounding this
queue instead would block the muxing thread inside a C write callback, and the
three places that reach that callback (``mux()``, the muxer's closing flush,
and ``RtspRelayStream.aclose()``) would each hang a shutdown and hold the RTSP
socket open behind it.
"""

from __future__ import annotations

import io
import queue
import threading

__all__ = ["PipeBuffer"]


class PipeBuffer(io.RawIOBase):
    """A blocking queue written by one thread and read by another."""

    _EOF = object()

    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.buffer: bytes = b""
        self._at_eof = False
        # Bytes written and not yet handed back, so the relay loop above can
        # see the upload falling behind and shed load before memory does.
        # Guarded rather than a bare int: the writer and the reader are
        # different threads and `+=` is not one operation.
        self._pending_lock = threading.Lock()
        self._pending = 0

    @property
    def pending_bytes(self) -> int:
        """Bytes written but not yet read.

        The measure of how far behind the upload is. Read by the relay loop
        every packet, which is why it is a counter rather than a walk of the
        queue: `queue.Queue` knows how many items it holds, not how big they
        are, and the muxer's writes are whatever avio happened to flush.
        """
        with self._pending_lock:
            return self._pending

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        if isinstance(b, str):
            b = b.encode("utf-8")
        # Counted before the put, not after: a reader fast enough to take the
        # chunk in between would otherwise decrement first and leave the count
        # briefly negative, which is the one moment the relay loop might read
        # it and conclude the upload is keeping up.
        with self._pending_lock:
            self._pending += len(b)
        self.queue.put(b)
        return len(b)

    def signal_eof(self) -> None:
        """No more writes are coming.

        Not an override of ``close()``: that marks the object closed, and the
        reader is a different thread still draining what is already queued.
        """
        self.queue.put(self._EOF)

    def readinto(self, b) -> int:
        """Fill ``b`` from the queue, blocking until there is something to give.

        ``readinto`` rather than ``read``, so ``RawIOBase`` derives ``read`` and
        ``readall`` from it and they behave the way the io contract says: a
        zero-length read returns immediately instead of waiting for data that
        was never asked for, and a read of everything keeps going past chunk
        boundaries to the end of the stream rather than stopping at the first.
        Writing ``read`` by hand gets one of those wrong, which is how both were
        wrong here.
        """
        if len(b) == 0:
            return 0

        # Loops rather than tests once: a zero-length write is legitimate - a
        # muxer flushing nothing produces one - and returning 0 for it tells
        # RawIOBase.readall that the stream ended, silently discarding
        # everything written afterwards.
        while not self.buffer:
            if self._at_eof:
                return 0
            # Blocks until data is available.
            chunk = self.queue.get(block=True, timeout=None)
            if chunk is self._EOF:
                self._at_eof = True
                return 0
            self.buffer = chunk

        taken = min(len(b), len(self.buffer))
        b[:taken] = self.buffer[:taken]
        self.buffer = self.buffer[taken:]
        # Decremented on the hand-off, not when the chunk left the queue: what
        # is sitting in `self.buffer` is still memory this object is holding.
        with self._pending_lock:
            self._pending -= taken
        return taken
