import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import mimetypes
import pathlib
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import aiohttp

from eyepop.auth import ExchangeError, exchange

log = logging.getLogger("eyepop.pipeline")

_PATH = "/api/v1/pipeline"
_REFRESH_SECONDS_BEFORE_EXPIRY = 60.0
_CHUNK = 64 * 1024
_NO_CAPACITY = ("no_pipeline_available", "overloaded")
_REOPEN_SECONDS = 1.0

Image = Mapping[str, str] | bytes | pathlib.Path | BinaryIO
Media = bytes | pathlib.Path | BinaryIO | AsyncIterable[bytes]


class EyePopError(Exception):
    """[EXPERIMENTAL] Any field that did not arrive is None (`message` is "")."""

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
    def __init__(self, wait: float, error: EyePopError):
        self.wait = wait
        self.error = error


class EyePopClient:
    """A client for the pipeline API [EXPERIMENTAL].

    `url` is the pipeline API's base url. Pass exactly one credential: `access_token`, a JWT, or
    `api_key`, which the client exchanges at `compute_url` and never sends to the pipeline API. It
    reads no environment variable. `heartbeat` is the socket's ping interval in seconds; a socket
    that sends no pong within half of it is dead. `wait` is how many seconds a call keeps retrying
    while the pipeline API has no capacity for it: a `503` with Retry-After, or a socket refused
    with `no_pipeline_available` or `overloaded` before it yielded anything.

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
        wait: float = 30.0,
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
        self._wait = wait
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._sockets: set[aiohttp.ClientWebSocketResponse] = set()

    async def __aenter__(self) -> "EyePopClient":
        """Return the client."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the client."""
        await self.close()

    async def close(self) -> None:
        while self._sockets:
            await self._sockets.pop().close()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def image(
        self,
        image: Image,
        *,
        ability: str | None = None,
        pop: Mapping[str, Any] | None = None,
        model: str | None = None,
        content_type: str | None = None,
    ) -> list[Any]:
        """Run one image through a pop and return its `predictions` as the pipeline wrote them.

        `image` is `{"url": ...}` or `{"assetUuid": ...}`, sent as JSON, or the image's bytes,
        a `pathlib.Path` or a binary file, sent as multipart and typed from the file's name or
        `content_type`. Pass exactly one of `ability` (an ability name), `pop` (a pop's JSON) or
        `model` (a model's uuid).
        """
        request = await _ImageRequest.read(image, _pop(ability, pop, model), content_type)
        return await self._retrying(lambda token: self._post(request, token), time.monotonic() + self._wait)

    async def socket(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        ability: str | None = None,
        pop: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open one socket, send the request, and yield each `start`, `prediction`, `dropped` and `end`.

        `items` are the WebSocket contract's items with the caller's ids. An upload item holds its
        media under `file` (bytes, a `pathlib.Path`, a binary file or an async iterable of bytes),
        sent after the server's `start` for it. Iteration ends after `done`; stopping it, or closing
        the client, closes the socket with `1000`. Any other close or failure raises `EyePopError`.
        Pass exactly one of `ability`, `pop` and `model`, as for `image`. A socket refused for want
        of capacity before it yielded anything is reopened with the same request until `wait` passes.
        """
        request = {"pop": _pop(ability, pop, model), "items": [{k: v for k, v in item.items() if k != "file"} for item in items]}
        media = {item["id"]: item["file"] for item in items if "file" in item}
        deadline = time.monotonic() + self._wait
        while True:
            yielded = False
            try:
                async with contextlib.aclosing(self._socket_once(request, media, deadline)) as messages:
                    async for message in messages:
                        yielded = True
                        yield message
                return
            except EyePopError as e:
                # nothing yielded means no upload started either, so the request can be resent whole
                if yielded or e.error not in _NO_CAPACITY or time.monotonic() + _REOPEN_SECONDS > deadline:
                    raise
                log.debug("the pipeline API refused the socket with %s, reopening in %ss", e.error, _REOPEN_SECONDS)
            await asyncio.sleep(_REOPEN_SECONDS)

    async def _socket_once(
        self, request: dict[str, Any], media: dict[Any, Media], deadline: float
    ) -> AsyncGenerator[dict[str, Any]]:
        ws = await self._retrying(self._open_socket, deadline)
        self._sockets.add(ws)
        uploads = _Uploads(ws, media)
        error: dict[str, Any] | None = None
        try:
            await ws.send_json(request)
            while (msg := await ws.receive()).type == aiohttp.WSMsgType.TEXT:
                try:
                    message = json.loads(msg.data)
                except ValueError:
                    message = None
                if not isinstance(message, dict):
                    raise EyePopError("the pipeline API sent a message that is not a JSON object")
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

    async def _valid_token(self, rejected: str | None = None) -> str:
        """One exchange runs at a time: callers that waited on it take its token instead of exchanging again.

        `rejected` is a token the pipeline API answered 401; it is exchanged unless another caller already has.
        """
        if self._api_key is None:
            assert self._access_token is not None
            return self._access_token
        seen = self._access_token
        async with self._lock:
            if self._access_token == seen and (
                seen is None or seen == rejected or self._expires - time.monotonic() < _REFRESH_SECONDS_BEFORE_EXPIRY
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
        try:
            expires = time.monotonic() + float(token.expires_in)
        except (TypeError, ValueError):
            raise EyePopError("compute-api answered the API key exchange with no valid expires_in") from None
        self._access_token = token.access_token
        self._expires = expires

    def _client_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _retrying[T](self, attempt: Callable[[str], Awaitable[T]], deadline: float) -> T:
        rejected: str | None = None
        while True:
            token = await self._valid_token(rejected)
            try:
                return await attempt(token)
            except EyePopError as e:
                if e.status != 401 or self._api_key is None or rejected is not None:
                    raise
                log.debug("the pipeline API answered 401, exchanging the API key again")
                rejected = token
            except _RetryAfter as answer:
                if time.monotonic() + answer.wait > deadline:
                    raise answer.error from None
                log.debug("the pipeline API answered %s, retrying in %ss", answer.error, answer.wait)
                await asyncio.sleep(answer.wait)

    async def _post(self, request: "_ImageRequest", token: str) -> list[Any]:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with self._client_session().post(self._url, headers=headers, **request.body()) as response:
                if response.status == 200:
                    try:
                        return (await response.json(content_type=None))["predictions"]
                    except (ValueError, KeyError, TypeError) as e:
                        raise EyePopError("the pipeline API answered 200 with no predictions", status=200) from e
                error = _answer_error(response.status, response.content_type, await response.text())
                wait = _retry_after(response.status, response.headers)
        except (aiohttp.ClientError, OSError, TimeoutError) as e:
            raise _unreachable(e) from e
        if wait is not None:
            raise _RetryAfter(wait, error)
        raise error

    async def _open_socket(self, token: str) -> aiohttp.ClientWebSocketResponse:
        headers = {"Authorization": f"Bearer {token}"}
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
        # `uploaded` ends the server's draining of an item that ended before its bytes did
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
    """Plain `http` only to a loopback host, so a token never crosses a network in clear."""
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


def _pop(ability: str | None, pop: Mapping[str, Any] | None, model: str | None) -> Any:
    if [ability, pop, model].count(None) != 2:
        raise ValueError("pass exactly one of ability, pop and model")
    if pop is not None:
        return pop
    component = {"ability": ability} if ability is not None else {"modelUuid": model}
    return {"components": [{"type": "inference", **component}]}


def _retry_after(status: int, headers: Mapping[str, str] | None) -> float | None:
    if status != 503 or not headers:
        return None
    try:
        wait = float(headers["Retry-After"])
    except (KeyError, ValueError):
        return None
    # a floor, so that `Retry-After: 0` cannot turn an overload into a request storm
    return max(wait, 1.0) if math.isfinite(wait) and wait >= 0 else None


def _answer_error(status: int, content_type: str, text: str) -> EyePopError:
    """`401` and `403` answer plain text, kept as the message."""
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
