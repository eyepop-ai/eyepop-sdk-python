from typing import Any, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from aiohttp import _RequestContextManager


class ClientSession:
    async def request_with_retry(
        self,
        method: str,
        url: str,
        accept: str | None = None,
        data: Any = None,
        content_type: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> "_RequestContextManager":
        raise NotImplementedError
