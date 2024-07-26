from typing import Callable, Any

import aiohttp


class WorkerClientSession:
    async def pipeline_get(
            self, url_path_and_query: str,
            accept: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass

    async def pipeline_post(
            self, url_path_and_query: str,
            accept: str | None = None,
            open_data: Callable = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass

    async def pipeline_patch(
            self, url_path_and_query: str,
            accept: str | None = None,
            data: Any = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass

    async def pipeline_delete(
            self, url_path_and_query: str,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        pass
