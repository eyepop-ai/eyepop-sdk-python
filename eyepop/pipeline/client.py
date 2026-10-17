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
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import aiohttp

from eyepop.auth import ExchangeError, exchange

log = logging.getLogger("eyepop.pipeline")

_PATH = "/api/v1/pipeline"
_REFRESH_BEFORE = 60.0  # seconds of a token's life left when the client exchanges again
_CHUNK = 64 * 1024

Image = Mapping[str, str] | bytes | pathlib.Path | BinaryIO
Media = bytes | pathlib.Path | BinaryIO | AsyncIterable[bytes]
Ability = str | Mapping[str, Any]


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


class _RetryAfter(Exception):
    """A `503` the contracts say to retry once `wait` seconds have passed."""

    def __init__(self, wait: float, error: EyePopError):
        self.wait = wait
        self.error = error


class EyePopClient:
    """A client for the pipeline API [EXPERIMENTAL].

    `url` is the pipeline API's base url. Pass exactly one credential: `access_token`, a JWT, or
    `api_key`, which the client exchanges at `compute_url` and never sends to the pipeline API. It
    reads no environment variable. `heartbeat` is the socket's ping interval in seconds; a socket
    that sends no pong within half of it is dead.

        async with EyePopClient(url, api_key=key) as client:
            predictions = await client.image(pathlib.Path("people.jpg"), ability="eyepop.person")
    """

    _retry_bound = 30.0  # seconds after the first try that a 503 stops being retried

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
        self._url = _checked_url(url, "url") + _PATH
        self._ws_url = "ws" + self._url.removeprefix("http")
        self._compute_url = _checked_url(compute_url, "compute_url")
        self._api_key = api_key
        self._access_token = access_token
        self._expires = 0.0
        self._heartbeat = heartbeat
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._sockets: set[aiohttp.ClientWebSocketResponse] = set()

    async def __aenter__(self) -> "EyePopClient":
        """Return the client; leaving the block closes it."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the client."""
        await self.close()

    async def close(self) -> None:
        """Close every open socket with `1000`, then the client's connections."""
        while self._sockets:
            await self._sockets.pop().close()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def image(self, image: Image, ability: Ability, content_type: str | None = None) -> list[Any]:
        """Run one image through a pop and return its `predictions` as the pipeline wrote them.

        `image` is `{"url": ...}` or `{"assetUuid": ...}`, sent as JSON, or the image's bytes,
        a `pathlib.Path` or a binary file, sent as multipart and typed from the file's name or
        `content_type`. `ability` is an ability name or a pop's JSON.
        """
        request = await _ImageRequest.read(image, _pop(ability), content_type)
        return await self._retrying(lambda: self._post(request))

    async def socket(self, items: list[Mapping[str, Any]], ability: Ability) -> AsyncIterator[dict[str, Any]]:
        """Open one socket, send the request, and yield each `start`, `prediction`, `dropped` and `end`.

        `items` are the WebSocket contract's items with the caller's ids. An upload item holds its
        media under `file` (bytes, a `pathlib.Path`, a binary file or an async iterable of bytes),
        sent after the server's `start` for it. Iteration ends after `done`; stopping it, or closing
        the client, closes the socket with `1000`. Any other close or failure raises `EyePopError`.
        """
        request = {"pop": _pop(ability), "items": [{k: v for k, v in item.items() if k != "file"} for item in items]}
        ws = await self._retrying(self._open_socket)
        self._sockets.add(ws)
        uploads = _Uploads(ws, {item["id"]: item["file"] for item in items if "file" in item})
        error: dict[str, Any] | None = None
        try:
            await ws.send_json(request)
            while (msg := await ws.receive()).type == aiohttp.WSMsgType.TEXT:
                message = json.loads(msg.data)
                await uploads.follow(message)
                if message.get("type") == "error":
                    error = message
                elif message.get("type") != "done":
                    yield message
            failure = uploads.failure or _socket_failure(msg, error, closed_by_client=ws not in self._sockets)
            if failure is not None:
                raise failure
        finally:
            self._sockets.discard(ws)
            uploads.cancel()
            await ws.close()

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
                await self._exchange(self._api_key)
            assert self._access_token is not None
            return self._access_token

    async def _exchange(self, api_key: str) -> None:
        try:
            token = await exchange(self._client_session(), self._compute_url, api_key)
        except ExchangeError as e:
            raise EyePopError(e.message, status=e.status) from e
        if not token.access_token:
            raise EyePopError("compute-api answered the API key exchange with no access_token")
        self._access_token = token.access_token
        self._expires = time.monotonic() + token.expires_in

    async def _authorization(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {await self._valid_token()}"}

    def _client_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _retrying[T](self, attempt: Callable[[], Awaitable[T]]) -> T:
        start = time.monotonic()
        while True:
            try:
                return await attempt()
            except _RetryAfter as answer:
                if time.monotonic() + answer.wait - start > self._retry_bound:
                    raise answer.error from None
                log.debug("the pipeline API answered %s, retrying in %ss", answer.error, answer.wait)
                await asyncio.sleep(answer.wait)

    async def _post(self, request: "_ImageRequest") -> list[Any]:
        headers = await self._authorization()
        try:
            async with self._client_session().post(self._url, headers=headers, **request.body()) as response:
                if response.status == 200:
                    return (await response.json())["predictions"]
                error = _answer_error(response.status, response.content_type, await response.text())
                wait = _retry_after(response.status, response.headers)
        except (aiohttp.ClientError, OSError, TimeoutError) as e:
            raise _unreachable(e) from e
        if wait is not None:
            raise _RetryAfter(wait, error)
        raise error

    async def _open_socket(self) -> aiohttp.ClientWebSocketResponse:
        headers = await self._authorization()
        try:
            return await self._client_session().ws_connect(
                self._ws_url, protocols=("eyepop.v1",), headers=headers, heartbeat=self._heartbeat
            )
        except aiohttp.WSServerHandshakeError as e:
            # aiohttp hides a refused upgrade's body; the only 503 an upgrade answers is `overloaded`
            overloaded = "overloaded" if e.status == 503 else None
            error = EyePopError("the pipeline API refused the upgrade", status=e.status, error=overloaded)
            wait = _retry_after(e.status, e.headers)
            if wait is not None:
                raise _RetryAfter(wait, error) from e
            raise error from e
        except (aiohttp.ClientError, OSError, TimeoutError) as e:
            raise _unreachable(e) from e


@dataclass(frozen=True)
class _ImageRequest:
    """One image's `POST`, read once so that a retry resends the same body."""

    payload: dict[str, Any]
    image: bytes | None = None
    content_type: str = ""
    filename: str = "image"

    @classmethod
    async def read(cls, image: Image, pop: Any, content_type: str | None) -> "_ImageRequest":
        if isinstance(image, Mapping):
            return cls({"pop": pop, "source": dict(image)})
        if isinstance(image, bytes):
            name, data = None, image
        elif isinstance(image, pathlib.Path):
            name, data = image.name, await asyncio.to_thread(image.read_bytes)
        else:
            name, data = getattr(image, "name", None), await asyncio.to_thread(image.read)
        guessed = mimetypes.guess_type(name)[0] if name else None
        return cls({"pop": pop}, data, content_type or guessed or "application/octet-stream", name or "image")

    def body(self) -> dict[str, Any]:
        if self.image is None:
            return {"json": self.payload}
        form = aiohttp.FormData()
        form.add_field("request", json.dumps(self.payload), content_type="application/json")
        form.add_field("image", self.image, content_type=self.content_type, filename=self.filename)
        return {"data": form}


class _Uploads:
    """Sends each upload item's media once the server has sent its `start`, and stops at its `end`."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, media: dict[Any, Media]):
        self._ws = ws
        self._media = media
        self._senders: dict[Any, asyncio.Task[None]] = {}
        self._uploaded: set[Any] = set()
        self.failure: EyePopError | None = None

    async def follow(self, message: dict[str, Any]) -> None:
        kind, item_id = message.get("type"), message.get("id")
        if kind == "start" and item_id in self._media and item_id not in self._senders:
            self._senders[item_id] = asyncio.create_task(self._send(item_id))
        elif kind == "end" and item_id in self._senders:
            await self._stop(item_id)

    def cancel(self) -> None:
        for sender in self._senders.values():
            sender.cancel()

    async def _send(self, item_id: Any) -> None:
        try:
            async for chunk in _chunks(self._media[item_id]):
                await self._ws.send_bytes(chunk)
        except Exception as e:
            self.failure = EyePopError(f"sending the media of upload {item_id!r} failed")
            self.failure.__cause__ = e
            await self._ws.close()
            return
        await self._send_uploaded(item_id)

    async def _stop(self, item_id: Any) -> None:
        """An item that ends before its bytes do stops them, and its `uploaded` ends the server's draining."""
        sender = self._senders[item_id]
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender
        await self._send_uploaded(item_id)

    async def _send_uploaded(self, item_id: Any) -> None:
        if item_id not in self._uploaded:
            self._uploaded.add(item_id)
            await self._ws.send_json({"type": "uploaded", "id": item_id})


def _socket_failure(msg: aiohttp.WSMessage, error: dict[str, Any] | None, closed_by_client: bool) -> EyePopError | None:
    """Why a socket ended, or None when it ended as it should: `1000` with no `error`, or closed by the client."""
    if msg.type == aiohttp.WSMsgType.CLOSE:
        code, reason = msg.data, msg.extra or None
        if error is not None:
            return EyePopError(error.get("message", ""), close_code=code, error=error.get("error") or reason)
        if code == aiohttp.WSCloseCode.OK:
            return None
        return EyePopError(f"the pipeline API closed the socket with {code}", close_code=code, error=reason)
    if msg.type == aiohttp.WSMsgType.ERROR:
        no_pong = isinstance(msg.data, aiohttp.ServerTimeoutError)
        failure = EyePopError("no pong within the heartbeat" if no_pong else "the connection failed", close_code=1006)
        failure.__cause__ = msg.data
        return failure
    if closed_by_client:
        return None
    return EyePopError("the connection dropped with no close frame", close_code=1006)


def _checked_url(url: str, name: str) -> str:
    """The url without a trailing slash; plain `http` only to a loopback host, so a token never crosses a network in clear."""
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


def _pop(ability: Ability) -> Any:
    if isinstance(ability, str):
        return {"components": [{"type": "inference", "ability": ability}]}
    return ability


def _retry_after(status: int, headers: Mapping[str, str] | None) -> float | None:
    if status != 503 or not headers:
        return None
    try:
        return float(headers["Retry-After"])
    except (KeyError, ValueError):
        return None


def _answer_error(status: int, content_type: str, text: str) -> EyePopError:
    """request.md's `{"error", "message"}` body; `401` and `403` are plain text, kept as the message."""
    if content_type == "application/json":
        with contextlib.suppress(ValueError):
            body = json.loads(text)
            if isinstance(body, dict):
                return EyePopError(body.get("message", ""), status=status, error=body.get("error"))
    return EyePopError(text.strip(), status=status)


def _unreachable(e: Exception) -> EyePopError:
    return EyePopError(f"could not reach the pipeline API: {type(e).__name__}: {e}")


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
