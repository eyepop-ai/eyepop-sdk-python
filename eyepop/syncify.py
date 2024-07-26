import asyncio
import logging
import threading
import types
import typing

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
        _run_coro_thread_save(self.event_loop, self.endpoint.connect())

    def disconnect(self):
        _run_coro_thread_save(self.event_loop, self.endpoint.disconnect())

    def session(self) -> dict:
        return _run_coro_thread_save(self.event_loop, self.endpoint.session())

    def _run_event_loop(self, event_loop):
        log.debug("_run_event_loop start")
        asyncio.set_event_loop(event_loop)
        event_loop.run_forever()
        log.debug("_run_event_loop done")


def _run_coro_thread_save(event_loop, coro):
    try:
        result = asyncio.run_coroutine_threadsafe(coro, event_loop).result()
        coro = None
        return result
    finally:
        if coro is not None:
            coro.close()

