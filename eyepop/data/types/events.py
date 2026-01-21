"""Event types for the EyePop Data API."""

from typing import Awaitable, Callable

from pydantic import BaseModel

from eyepop.data.types.enums import ChangeType


class ChangeEvent(BaseModel):
    change_type: ChangeType
    account_uuid: str
    dataset_uuid: str | None
    dataset_version: int | None
    asset_uuid: str | None
    mdl_uuid: str | None
    workflow_id: str | None
    message: str | None
    workflow_task_name: str | None


EventHandler = Callable[[ChangeEvent], Awaitable[None]]
