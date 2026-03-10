import asyncio
from contextlib import suppress


class Periodic:
    def __init__(self, func, time: float):
        self.func = func
        self.time = time
        self.is_started = False
        self._task: asyncio.Task | None = None

    async def start(self):
        if not self.is_started:
            self.is_started = True
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self.is_started:
            self.is_started = False
            if self._task is not None and not self._task.done():
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
            self._task = None

    async def _run(self):
        while True:
            await asyncio.sleep(self.time)
            await self.func()
