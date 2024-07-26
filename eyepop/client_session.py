from typing import Any

import aiohttp


class ClientSession:
    async def request_with_retry(self, method: str, url: str, accept: str | None = None,
                                 data: Any = None, content_type: str | None = None,
                                 timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        raise NotImplementedError