import asyncio
from asyncio import Future, sleep
from typing import Callable, Awaitable

from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_types import ChangeEvent


class WaitFor:
    endpoint: DataEndpoint
    dataset_uuid: str
    future: Future | None
    criteria: Callable[[DataEndpoint, ChangeEvent], Awaitable[bool]]
    def __init__(self, endpoint: DataEndpoint, dataset_uuid: str, criteria: Callable[[DataEndpoint, ChangeEvent], Awaitable[bool]]):
        self.endpoint = endpoint
        self.dataset_uuid = dataset_uuid
        self.criteria = criteria
        self.future = None

    async def __aenter__(self) -> "WaitFor":
        self.future = asyncio.get_running_loop().create_future()
        await self.endpoint.add_dataset_event_handler(self.dataset_uuid, self._on_change_event)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        try:
            await self.future
        finally:
            await self.endpoint.remove_dataset_event_handler(self.dataset_uuid, self._on_change_event)

    async def _on_change_event(self, event: ChangeEvent) -> None:
        if self.future.done():
            return
        try:
            if await self.criteria(self.endpoint, event):
                self.future.set_result(None)
        except BaseException as e:
            self.future.set_exception(e)
