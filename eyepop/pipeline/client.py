import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import mimetypes
import os
import pathlib
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import urlsplit

import aiohttp

from eyepop.auth import ExchangeError, exchange
from eyepop.pipeline.pop import Ability, Model, Pop
from eyepop.pipeline.prediction import Prediction, Result

log = logging.getLogger("eyepop.pipeline")

_PATH = "/api/v1/pipeline"
_REFRESH_SECONDS_BEFORE_EXPIRY = 60.0
_CHUNK = 64 * 1024
_NO_CAPACITY = ("pipeline_start_failed", "overloaded")
_REOPEN_SECONDS = 1.0

Image = str | Mapping[str, str] | bytes | pathlib.Path | BinaryIO
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


class EyePop:
    """A client for the pipeline API [EXPERIMENTAL].

    `url` is the pipeline API's base url. Pass exactly one credential: `access_token`, a JWT, or
    `api_key`, which the client exchanges at `compute_url` and never sends to the pipeline API. The
    constructor reads no environment variable; `EyePop.init()` does. `heartbeat` is the socket's
    ping interval in seconds; a socket that sends no pong within half of it is dead. `wait` is how
    many seconds a call keeps retrying while the pipeline API has no capacity for it: a `503` with
    Retry-After, or a socket refused with `pipeline_start_failed` or `overloaded` before it
    yielded anything.

        async with EyePop.init() as ep:
            person = ep.use(Ability("eyepop.person"))
            prediction = await person.image("people.jpg")
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

    @classmethod
    def init(
        cls,
        url: str | None = None,
        access_token: str | None = None,
        api_key: str | None = None,
        compute_url: str = "https://compute.eyepop.ai",
        heartbeat: float = 20.0,
        wait: float = 30.0,
    ) -> "EyePop":
        """A client that takes what is not passed from the environment, or from `.env` in the working directory.

        It reads `EYEPOP_PIPELINE_URL`, and `EYEPOP_API_KEY` or else `EYEPOP_ACCESS_TOKEN`; the
        environment wins over `.env`. Nothing is sent before the first call.
        """
        env = {**_dotenv(pathlib.Path(".env")), **os.environ}
        url = url or env.get("EYEPOP_PIPELINE_URL")
        if not url:
            raise ValueError("pass url or set EYEPOP_PIPELINE_URL")
        if access_token is None and api_key is None:
            api_key = env.get("EYEPOP_API_KEY") or None
            access_token = None if api_key else env.get("EYEPOP_ACCESS_TOKEN") or None
        return cls(url, access_token, api_key, compute_url, heartbeat, wait)

    async def __aenter__(self) -> "EyePop":
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

    def use(self, pop: "Ability | Model | Pop") -> "Runner":
        """A runner of `pop`: an ability, a model, either one with `.crop(...)`, or a pop's JSON."""
        return Runner(self, pop.json)

    async def _image(self, pop: Mapping[str, Any], image: Image, content_type: str | None) -> Prediction:
        request = await _ImageRequest.read(image, pop, content_type)
        return await self._retrying(lambda token: self._post(request, token), time.monotonic() + self._wait)

    async def _socket(self, pop: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> AsyncGenerator[dict[str, Any]]:
        """Open one socket, send the request, and yield each `start`, `prediction`, `dropped` and `end`.

        `items` are the WebSocket contract's items. An upload item holds its media under `file`,
        sent after the server's `start` for it. Iteration ends after `done`; stopping it, or closing
        the client, closes the socket with `1000`. Any other close or failure raises `EyePopError`.
        A socket refused for want of capacity before it yielded anything is reopened with the same
        request until `wait` passes.
        """
        request = {"pop": pop, "items": [{k: v for k, v in item.items() if k != "file"} for item in items]}
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

    async def _post(self, request: "_ImageRequest", token: str) -> Prediction:
        headers = {"Authorization": f"Bearer {token}"}
        try:
            async with self._client_session().post(self._url, headers=headers, **request.body()) as response:
                if response.status == 200:
                    try:
                        return Prediction.parse((await response.json(content_type=None))["predictions"][0])
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
                        raise EyePopError("the pipeline API answered 200 with no prediction", status=200) from e
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


class Runner:
    """[EXPERIMENTAL] Runs one pop; from `EyePop.use`.

    An image is a url (a `str` with `://`), a path (any other `str`, or a `pathlib.Path`), bytes, a
    binary file, or the request's `source` as a mapping, such as `{"assetUuid": ...}`. A path or a
    file is typed from its name, bytes from `content_type`.
    """

    def __init__(self, client: EyePop, pop: Mapping[str, Any]):
        self._client = client
        self._pop = pop

    async def image(self, image: Image, content_type: str | None = None) -> Prediction:
        """One image's prediction, over one `POST`."""
        return await self._client._image(self._pop, image, content_type)

    async def batch(self, images: Iterable[Image]) -> AsyncIterator[Result]:
        """Yield each image's prediction, or its error when it failed, as it arrives over one socket."""
        sources = {str(i): image for i, image in enumerate(images)}
        messages = self._client._socket(self._pop, [_item(i, image) for i, image in sources.items()])
        async with contextlib.aclosing(messages):
            async for message in messages:
                source = sources.get(str(message.get("id")))
                if message.get("type") == "prediction":
                    yield Result(source, _prediction(message.get("prediction")))
                elif _failed(message):
                    yield Result(source, error=message.get("message") or "the item failed")

    def video(self, video: str | Media, content_type: str | None = None) -> AsyncIterator[Prediction]:
        """Yield each frame's prediction of a video: a url, a path, bytes, a binary file or an async iterable of bytes."""
        return self._frames(_item("video", video, content_type))

    def stream(self, url: str) -> AsyncIterator[Prediction]:
        """Yield each frame's prediction of a live stream until the caller stops iterating or closes the client."""
        return self._frames({"id": "stream", "url": url, "live": True})

    async def _frames(self, item: dict[str, Any]) -> AsyncIterator[Prediction]:
        messages = self._client._socket(self._pop, [item])
        async with contextlib.aclosing(messages):
            async for message in messages:
                if message.get("type") == "prediction":
                    yield _prediction(message.get("prediction"))
                elif _failed(message):
                    raise EyePopError(message.get("message") or "the item failed")


@dataclass(frozen=True)
class _ImageRequest:
    """One image's `POST`, read once so that a retry resends the same body."""

    payload: dict[str, Any]
    image: bytes | None = None
    content_type: str = ""
    filename: str = "image"

    @classmethod
    async def read(cls, image: Image, pop: Any, content_type: str | None) -> "_ImageRequest":
        if isinstance(image, str):
            image = _from_str(image)
        if isinstance(image, Mapping):
            return cls({"pop": pop, "source": dict(image)})
        if isinstance(image, bytes):
            name, data = None, image
        elif isinstance(image, pathlib.Path):
            name, data = image.name, await asyncio.to_thread(image.read_bytes)
        else:
            name, data = getattr(image, "name", None), await asyncio.to_thread(image.read)
        return cls({"pop": pop}, data, _content_type(name, content_type), name or "image")

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


def _from_str(media: str) -> dict[str, str] | pathlib.Path:
    return {"url": media} if "://" in media else pathlib.Path(media)


def _content_type(name: object, content_type: str | None) -> str:
    # a file opened from a descriptor is named by an int
    guessed = mimetypes.guess_type(name)[0] if isinstance(name, str) else None
    return content_type or guessed or "application/octet-stream"


def _item(item_id: str, media: Image | Media, content_type: str | None = None) -> dict[str, Any]:
    if isinstance(media, str):
        media = _from_str(media)
    if isinstance(media, Mapping):
        return {"id": item_id, **media}
    name = None if isinstance(media, bytes) else getattr(media, "name", None)
    return {"id": item_id, "upload": {"contentType": _content_type(name, content_type)}, "file": media}


def _failed(message: dict[str, Any]) -> bool:
    return message.get("type") == "end" and message.get("reason") == "error"


def _prediction(data: Any) -> Prediction:
    try:
        return Prediction.parse(data)
    except (AttributeError, TypeError, ValueError) as e:
        raise EyePopError("the pipeline API sent a prediction that is not one") from e


def _dotenv(path: pathlib.Path) -> dict[str, str]:
    # shortcut: KEY=value lines only, no inline comments, expansion or multi-line values; python-dotenv if that is needed
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    env = {}
    for line in lines:
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        if sep and not key.startswith("#"):
            env[key.strip()] = value.strip().strip("\"'")
    return env


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
