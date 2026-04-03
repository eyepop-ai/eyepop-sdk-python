from typing import Any, Callable

import aiohttp

from eyepop.client_session import ClientSession


class WorkerClientSession(ClientSession):
    async def pipeline_get(
            self, url_path_and_query: str,
            accept: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> aiohttp.ClientResponse:
        raise NotImplementedError

    async def pipeline_post(
            self, url_path_and_query: str,
            accept: str | None = None,
            open_data: Callable | None = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> aiohttp.ClientResponse:
        raise NotImplementedError

    async def pipeline_patch(
            self, url_path_and_query: str,
            accept: str | None = None,
            data: Any = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> aiohttp.ClientResponse:
        raise NotImplementedError

    async def pipeline_delete(
            self, url_path_and_query: str,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> aiohttp.ClientResponse:
        raise NotImplementedError
