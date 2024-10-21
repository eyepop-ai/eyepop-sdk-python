from typing import Callable

from threading import Condition

from eyepop.data.data_syncify import SyncDataEndpoint
from eyepop.data.data_types import ChangeEvent


class WaitForSync:
    endpoint: SyncDataEndpoint
    dataset_uuid: str
    condition: Condition | None
    criteria: Callable[[SyncDataEndpoint, ChangeEvent], bool]
    result: any
    exception: any

    def __init__(self, endpoint: SyncDataEndpoint, dataset_uuid: str, criteria: Callable[[SyncDataEndpoint, ChangeEvent], bool]):
        self.endpoint = endpoint
        self.dataset_uuid = dataset_uuid
        self.criteria = criteria
        self.condition = None

    def __enter__(self) -> "SyncWaitFor":
        self.condition = Condition()
        self.result = None
        self.exception = None
        self.endpoint.add_dataset_event_handler(self.dataset_uuid, self._on_change_event)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            with self.condition:
                if self.result is None and self.exception is None:
                    self.condition.wait()
                if self.exception is not None:
                    raise self.exception
        finally:
            self.endpoint.remove_dataset_event_handler(self.dataset_uuid, self._on_change_event)

    def _on_change_event(self, event: ChangeEvent) -> None:
        if self.condition is None or self.result is not None or self.exception is not None:
            return
        try:
            if self.criteria(self.endpoint, event):
                with self.condition:
                    self.result = True
                    self.condition.notify_all()
        except BaseException as e:
            with self.condition:
                self.exception = e
                self.condition.notify_all()
