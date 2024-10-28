import asyncio
import io
import logging
import threading
import types
import typing
from asyncio import StreamReader

log = logging.getLogger(__name__)


class SyncEndpoint:
    def __init__(self, endpoint: "Endpoint"):
        self._on_ready = None
        self.endpoint = endpoint
        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, args=(self.event_loop,), daemon=True)
        self.thread.start()

    def __del__(self):
        self.event_loop.close()

    def __enter__(self) -> "SyncEndpoint":
        self.connect()
        return self

    def __exit__(
            self,
            exc_type: typing.Optional[typing.Type[BaseException]],
            exc_val: typing.Optional[BaseException],
            exc_tb: typing.Optional[types.TracebackType],
    ) -> None:
        self.disconnect()

    def connect(self):
        run_coro_thread_save(self.event_loop, self.endpoint.connect())

    def disconnect(self, timeout: float | None = None):
        run_coro_thread_save(self.event_loop, self.endpoint.disconnect(timeout))

    def session(self) -> dict:
        return run_coro_thread_save(self.event_loop, self.endpoint.session())

    def _run_event_loop(self, event_loop):
        log.debug("_run_event_loop start")
        asyncio.set_event_loop(event_loop)
        event_loop.run_forever()
        log.debug("_run_event_loop done")

    def _async_reader_to_sync_binary_io(self, async_stream_reader):
        queue = run_coro_thread_save(
            self.event_loop, _create_queue()
        )
        submit_coro_thread_save(
            self.event_loop, _drain_stream_reader_into_queue(async_stream_reader, queue)
        )
        sync_io = _async_queue_to_stream(self.event_loop, queue)
        return sync_io


def run_coro_thread_save(event_loop, coro):
    try:
        result = asyncio.run_coroutine_threadsafe(coro, event_loop).result()
        coro = None
        return result
    finally:
        if coro is not None:
            coro.close()


def submit_coro_thread_save(event_loop, coro):
    try:
        future = asyncio.run_coroutine_threadsafe(coro, event_loop)
        coro = None
        future.add_done_callback(lambda f: f.result())
    finally:
        if coro is not None:
            coro.close()


async def _create_queue() -> asyncio.Queue:
    return asyncio.Queue(maxsize=128)


async def _drain_stream_reader_into_queue(stream_reader: StreamReader, queue: asyncio.Queue):
    try:
        n = 0
        while True:
            buffer = await stream_reader.read(4096)
            if not buffer:
                break
            n += len(buffer)
            await queue.put(buffer)
    except Exception as e:
        await queue.put(e)
    finally:
        await queue.put(None)


# https://discuss.python.org/t/asynchronous-generator-to-io-bufferedreader/20503
# iterate over the request.stream(),
# putting the blocks into a queue
# have a separate thread that consumes the queue yielding bytes
# convert the generator into a buffered reader as per below>
# pass the buffered reader into tarfile.open(fileobj=...)

def _async_queue_to_stream(event_loop, queue: asyncio.Queue):
    class GeneratorStream(io.RawIOBase):
        def __init__(self):
            self.leftover = None
            self.eof = False

        def readable(self):
            return True

        def readinto(self, b):
            _l = len(b)  # : We're supposed to return at most this much
            chunk = self.leftover
            if not chunk and not self.eof:
                chunk = asyncio.run_coroutine_threadsafe(queue.get(), event_loop).result()
            if not chunk:
                self.eof = True
                return 0
            output, self.leftover = chunk[:_l], chunk[_l:]
            b[:len(output)] = output
            return len(output)
    return io.BufferedReader(GeneratorStream())
