"""A client for the pipeline API [EXPERIMENTAL].

It speaks eyepop-instance's `specs/012-pipeline-api/contracts/request.md` (one image, `POST`) and
`websocket.md` (batches, uploads and streams), and adds nothing to them.
"""

import asyncio
import contextlib
import ipaddress
import json
import logging
import mimetypes
import pathlib
import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import aiohttp

from eyepop.auth import ExchangeError, exchange

log = logging.getLogger("eyepop.pipeline")

_PATH = "/api/v1/pipeline"
_REFRESH_BEFORE = 60.0  # seconds of a token's life left when the client exchanges again
_RETRY_BOUND = 30.0  # seconds after the first try that a 503 with Retry-After stops being retried
_CHUNK = 64 * 1024

Media = bytes | pathlib.Path | BinaryIO | AsyncIterable[bytes]


class EyePopError(Exception):
    """Every error the pipeline API's contracts name [EXPERIMENTAL].

    `status` is the HTTP status, `close_code` the WebSocket close code, `error` the contract's code
    and `message` its text; whatever did not arrive is None (`message` is "").
    """

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        close_code: int | None = None,
        error: str | None = None,
    ):
        self.status = status
        self.close_code = close_code
        self.error = error
        self.message = message
        parts = [
            f"HTTP {status}" if status is not None else "",
            f"close {close_code}" if close_code is not None else "",
            error or "",
        ]
        head = " ".join(p for p in parts if p)
        super().__init__(f"{head}: {message}" if head and message else head or message)


def _check_url(url: str, name: str) -> str:
    """The url without a trailing slash; plain `http` only to a loopback host."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{name} must be an http or https url")
    if parts.scheme == "http" and not _is_loopback(parts.hostname):
        raise ValueError(f"{name} must use https unless its host is a loopback address")
    return url.rstrip("/")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _pop(ability: str | Mapping[str, Any]) -> Any:
    if isinstance(ability, str):
        return {"components": [{"type": "inference", "ability": ability}]}
    return ability


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    try:
        return float(headers["Retry-After"]) if headers else None
    except (KeyError, ValueError):
        return None


class EyePopClient:
    """A client for the pipeline API [EXPERIMENTAL].

    `url` is the pipeline API's base url. Pass exactly one credential: `access_token`, a JWT, or
    `api_key`, which the client exchanges at `compute_url` and never sends to the pipeline API. It
    reads no environment variable. `heartbeat` is the socket's ping interval in seconds; a socket
    that sends no pong within half of it is dead.

        async with EyePopClient(url, api_key=key) as client:
            predictions = await client.image(pathlib.Path("people.jpg"), ability="eyepop.person")
    """

    def __init__(
        self,
        url: str,
        access_token: str | None = None,
        api_key: str | None = None,
        compute_url: str = "https://compute.eyepop.ai",
        heartbeat: float = 20.0,
    ):
        if (access_token is None) == (api_key is None):
            raise ValueError("pass exactly one of access_token and api_key")
        self._url = _check_url(url, "url") + _PATH
        self._ws_url = "ws" + self._url[len("http"):]
        self._compute_url = _check_url(compute_url, "compute_url")
        self._api_key = api_key
        self._access_token = access_token
        self._expires = 0.0
        self._heartbeat = heartbeat
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._sockets: set[aiohttp.ClientWebSocketResponse] = set()

    async def __aenter__(self) -> "EyePopClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close every open socket with `1000`, then the client's connections."""
        while self._sockets:
            await self._sockets.pop().close()
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _client_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _valid_token(self, force: bool = False) -> str:
        """The JWT to send, exchanging the API key first when it is missing, near expiry, or `force`.

        One exchange runs at a time: callers that waited on it take its token instead of exchanging again.
        """
        if self._api_key is None:
            assert self._access_token is not None
            return self._access_token
        seen = self._access_token
        async with self._lock:
            if self._access_token == seen and (
                force or seen is None or self._expires - time.monotonic() < _REFRESH_BEFORE
            ):
                try:
                    token = await exchange(self._client_session(), self._compute_url, self._api_key)
                except ExchangeError as e:
                    raise EyePopError(e.message, status=e.status) from e
                if not token.access_token:
                    raise EyePopError("compute-api answered the API key exchange with no access_token")
                self._access_token = token.access_token
                self._expires = time.monotonic() + token.expires_in
            assert self._access_token is not None
            return self._access_token

    async def image(
        self,
        image: Mapping[str, str] | bytes | pathlib.Path | BinaryIO,
        ability: str | Mapping[str, Any],
        content_type: str | None = None,
    ) -> list[Any]:
        """Run one image through a pop and return its `predictions` as the pipeline wrote them.

        `image` is `{"url": ...}` or `{"assetUuid": ...}`, sent as JSON, or the image's bytes,
        a `pathlib.Path` or a binary file, sent as multipart and typed from the file's name or
        `content_type`. `ability` is an ability name or a pop's JSON.
        """
        pop = _pop(ability)
        body: bytes | None = None
        name: str | None = None
        if isinstance(image, Mapping):
            payload = {"pop": pop, "source": dict(image)}
        else:
            if isinstance(image, bytes):
                body = image
            elif isinstance(image, pathlib.Path):
                name = image.name
                body = await asyncio.to_thread(image.read_bytes)
            else:
                name = getattr(image, "name", None)
                body = await asyncio.to_thread(image.read)
            content_type = content_type or (name and mimetypes.guess_type(name)[0]) or "application/octet-stream"
            payload = {"pop": pop}

        def send() -> Any:
            if body is None:
                return payload
            form = aiohttp.FormData()
            form.add_field("request", json.dumps(payload), content_type="application/json")
            form.add_field("image", body, content_type=content_type, filename=name or "image")
            return form

        start = time.monotonic()
        while True:
            headers = {"Authorization": f"Bearer {await self._valid_token()}"}
            data = send()
            try:
                async with self._client_session().post(
                    self._url,
                    headers=headers,
                    json=data if body is None else None,
                    data=data if body is not None else None,
                ) as response:
                    if response.status == 200:
                        return (await response.json())["predictions"]
                    text = await response.text()
                    wait = _retry_after(response.headers) if response.status == 503 else None
                    error = _error_from_answer(response.status, response.content_type, text)
            except (aiohttp.ClientError, OSError, TimeoutError) as e:
                raise EyePopError(f"could not reach the pipeline API: {type(e).__name__}: {e}") from e
            if wait is None or time.monotonic() + wait - start > _RETRY_BOUND:
                raise error
            log.debug("the pipeline API answered %s %s, retrying in %ss", error.status, error.error, wait)
            await asyncio.sleep(wait)

    async def socket(
        self, items: list[Mapping[str, Any]], ability: str | Mapping[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Open one socket, send the request, and yield each `start`, `prediction`, `dropped` and `end`.

        `items` are the WebSocket contract's items with the caller's ids. An upload item holds its
        media under `file` (bytes, a `pathlib.Path`, a binary file or an async iterable of bytes),
        sent after the server's `start` for it. Iteration ends after `done`; stopping it, or closing
        the client, closes the socket with `1000`. Any other close or failure raises `EyePopError`.
        """
        media = {item["id"]: item["file"] for item in items if "file" in item}
        request = {"pop": _pop(ability), "items": [{k: v for k, v in item.items() if k != "file"} for item in items]}
        ws = await self._connect()
        self._sockets.add(ws)
        uploads: dict[Any, asyncio.Task[None]] = {}
        uploaded: set[Any] = set()
        failed: list[EyePopError] = []
        error: dict[str, Any] | None = None

        async def upload(item_id: Any) -> None:
            try:
                async for chunk in _chunks(media[item_id]):
                    await ws.send_bytes(chunk)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                failure = EyePopError(f"sending the media of upload {item_id!r} failed")
                failure.__cause__ = e
                failed.append(failure)
                await ws.close()
                return
            uploaded.add(item_id)
            await ws.send_json({"type": "uploaded", "id": item_id})

        try:
            await ws.send_json(request)
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    message = json.loads(msg.data)
                    kind = message.get("type")
                    item_id = message.get("id")
                    if kind == "start" and item_id in media and item_id not in uploads:
                        uploads[item_id] = asyncio.create_task(upload(item_id))
                    elif kind == "end" and item_id in uploads and item_id not in uploaded:
                        # the item ended before its bytes did: stop sending, and end the server's draining
                        uploads[item_id].cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await uploads[item_id]
                        if item_id not in uploaded:
                            uploaded.add(item_id)
                            await ws.send_json({"type": "uploaded", "id": item_id})
                    if kind == "error":
                        error = message
                    elif kind != "done":
                        yield message
                elif msg.type == aiohttp.WSMsgType.CLOSE:
                    code, reason = msg.data, msg.extra or None
                    if code == aiohttp.WSCloseCode.OK and error is None:
                        return
                    if error is not None:
                        raise EyePopError(error.get("message", ""), close_code=code, error=error.get("error") or reason)
                    raise EyePopError(f"the pipeline API closed the socket with {code}", close_code=code, error=reason)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    what = "no pong within the heartbeat" if isinstance(msg.data, aiohttp.ServerTimeoutError) else "the connection failed"
                    raise EyePopError(what, close_code=1006) from msg.data
                else:
                    if failed:
                        raise failed[0]
                    if ws not in self._sockets:
                        return  # the client was closed
                    raise EyePopError("the connection dropped with no close frame", close_code=1006)
        finally:
            self._sockets.discard(ws)
            for task in uploads.values():
                task.cancel()
            await ws.close()

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        start = time.monotonic()
        while True:
            headers = {"Authorization": f"Bearer {await self._valid_token()}"}
            try:
                return await self._client_session().ws_connect(
                    self._ws_url, protocols=("eyepop.v1",), headers=headers, heartbeat=self._heartbeat
                )
            except aiohttp.WSServerHandshakeError as e:
                wait = _retry_after(e.headers) if e.status == 503 else None
                if wait is None or time.monotonic() + wait - start > _RETRY_BOUND:
                    raise EyePopError(
                        "the pipeline API refused the upgrade", status=e.status, error="overloaded" if e.status == 503 else None
                    ) from e
                log.debug("the pipeline API answered the upgrade 503, retrying in %ss", wait)
                await asyncio.sleep(wait)
            except (aiohttp.ClientError, OSError, TimeoutError) as e:
                raise EyePopError(f"could not reach the pipeline API: {type(e).__name__}: {e}") from e


def _error_from_answer(status: int, content_type: str, text: str) -> EyePopError:
    """request.md's `{"error", "message"}` body; `401` and `403` may be plain text, kept as the message."""
    if content_type == "application/json":
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        if isinstance(body, dict):
            return EyePopError(body.get("message", ""), status=status, error=body.get("error"))
    return EyePopError(text.strip(), status=status)


async def _chunks(media: Media) -> AsyncIterator[bytes]:
    if isinstance(media, bytes):
        for i in range(0, len(media), _CHUNK):
            yield media[i : i + _CHUNK]
    elif isinstance(media, pathlib.Path):
        with await asyncio.to_thread(media.open, "rb") as f:
            while chunk := await asyncio.to_thread(f.read, _CHUNK):
                yield chunk
    elif isinstance(media, AsyncIterable):
        async for chunk in media:
            yield chunk
    else:
        while chunk := await asyncio.to_thread(media.read, _CHUNK):
            yield chunk
