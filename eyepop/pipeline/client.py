import asyncio
import contextlib
import ipaddress
import json
import logging
import math
import os
import pathlib
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from eyepop.auth import ExchangeError, exchange
from eyepop.pipeline.errors import EyePopError
from eyepop.pipeline.prediction import Prediction, Result
from eyepop.pipeline.session import (
    Image,
    Media,
    Pipeline,
    PopLike,
    _content_type,
    _pop_json,
)

log = logging.getLogger("eyepop.pipeline")

_PATH = "/api/v1/pipeline"
_COMPUTE_URL = "https://compute.eyepop.ai"
_REFRESH_SECONDS_BEFORE_EXPIRY = 60.0
_NO_CAPACITY = ("pipeline_start_failed", "overloaded")
_REOPEN_SECONDS = 1.0


class _RetryAfter(Exception):
    def __init__(self, wait: float, error: EyePopError):
        self.wait = wait
        self.error = error


class EyePop:
    """A client for the pipeline API [EXPERIMENTAL].

    `url` is the pipeline API's base url. Pass exactly one credential: `access_token`, a JWT, or
    `api_key`, which the client exchanges at `compute_url` and never sends to the pipeline API. The
    constructor reads no environment variable; `client()` does. `heartbeat` is the socket's
    ping interval in seconds; a socket that sends no pong within half of it is dead. `wait` is how
    many seconds a call keeps retrying while the pipeline API has no capacity for it: a `503` with
    Retry-After, or a socket refused with `pipeline_start_failed` or `overloaded` before it
    yielded anything.

        async with client() as ep:
            person = ep.use(Ability("eyepop.person"))
            prediction = await person.image("people.jpg")
    """

    def __init__(
        self,
        url: str,
        access_token: str | None = None,
        api_key: str | None = None,
        compute_url: str = _COMPUTE_URL,
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
        self._pipelines: set[Pipeline] = set()

    async def __aenter__(self) -> "EyePop":
        """Return the client."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the client."""
        await self.close()

    async def close(self) -> None:
        for pipeline in list(self._pipelines):
            await pipeline.close()
        if self._session is not None:
            await self._session.close()
            self._session = None

    def use(self, pop: PopLike) -> "Runner":
        """A runner of `pop`: an ability, a model, either one with `.crop(...)`, or a pop's JSON."""
        return Runner(self, pop)

    def pipeline(self, pop: PopLike | None = None) -> Pipeline:
        """A pipeline to keep: one socket you can change the pop of and set sources on. Open it with `async with`."""
        return Pipeline(self, pop)

    async def _image(self, pop: Mapping[str, Any], image: Image, content_type: str | None) -> Prediction:
        request = await _ImageRequest.read(image, pop, content_type)
        return await self._retrying(lambda token: self._post(request, token), time.monotonic() + self._wait)

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
    """[EXPERIMENTAL] Runs one pop; from `EyePop.use`. Each call opens a pipeline of its own and closes it.

    An image is a url (a `str` with `://`), a path (any other `str`, or a `pathlib.Path`), bytes, a
    binary file, or the request's `source` as a mapping, such as `{"assetUuid": ...}`. A path or a
    file is typed from its name, bytes from `content_type`. To keep a pipeline between calls, or to
    change its pop, use `EyePop.pipeline`.
    """

    def __init__(self, client: "EyePop", pop: PopLike):
        self._client = client
        self._pop = pop

    async def image(self, image: Image, content_type: str | None = None) -> Prediction:
        """One image's prediction, over one `POST`."""
        return await self._client._image(_pop_json(self._pop), image, content_type)

    def batch(self, images: Iterable[Image]) -> AsyncGenerator[Result]:
        """Yield each image's prediction, or its error when it failed, as it arrives over one socket."""
        return self._once(lambda p: p.batch(images))

    def video(self, video: str | Media, content_type: str | None = None) -> AsyncGenerator[Prediction]:
        """Yield each frame's prediction of a video: a url, a path, bytes, a binary file or an async iterable of bytes."""
        return self._once(lambda p: p.video(video, content_type).__aiter__())

    def stream(self, url: str) -> AsyncGenerator[Prediction]:
        """Yield each frame's prediction of a live stream until the caller stops iterating or closes the client."""
        return self._once(lambda p: p.stream(url).__aiter__())

    async def _once[T](self, work: Callable[[Pipeline], AsyncGenerator[T]]) -> AsyncGenerator[T]:
        """Run `work` on a pipeline of its own, reopened while the pipeline API has no capacity for it and nothing was yielded."""
        deadline = time.monotonic() + self._client._wait  # pyright: ignore[reportPrivateUsage]
        while True:
            yielded = False
            try:
                async with self._client.pipeline(self._pop) as pipeline, contextlib.aclosing(work(pipeline)) as items:
                    async for item in items:
                        yielded = True
                        yield item
                return
            except EyePopError as e:
                # nothing yielded means nothing was seen, so the call can run again whole
                if yielded or e.error not in _NO_CAPACITY or time.monotonic() + _REOPEN_SECONDS > deadline:
                    raise
                log.debug("the pipeline API had no capacity (%s), reopening in %ss", e.error, _REOPEN_SECONDS)
            await asyncio.sleep(_REOPEN_SECONDS)


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
        elif isinstance(image, AsyncIterable):
            raise TypeError("an image is a url, a path, bytes or a binary file")
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


_defaults: dict[str, Any] = {}


def configure(**options: Any) -> None:
    """[EXPERIMENTAL] Set what `client()` and the module's functions use for `url`, `api_key`, `access_token`, `compute_url`, `heartbeat` and `wait`. `None` unsets one."""
    for name, value in options.items():
        if value is None:
            _defaults.pop(name, None)
        else:
            _defaults[name] = value


def client(
    url: str | None = None,
    access_token: str | None = None,
    api_key: str | None = None,
    compute_url: str | None = None,
    heartbeat: float | None = None,
    wait: float | None = None,
) -> EyePop:
    """[EXPERIMENTAL] A client that takes what is not passed from `configure()`, the environment, or `.env` in the working directory.

    It reads `EYEPOP_PIPELINE_URL`, and `EYEPOP_API_KEY` or else `EYEPOP_ACCESS_TOKEN`; the
    environment wins over `.env`. Nothing is sent before the first call.
    """
    given = {
        "url": url,
        "access_token": access_token,
        "api_key": api_key,
        "compute_url": compute_url,
        "heartbeat": heartbeat,
        "wait": wait,
    }
    options = {**_defaults, **{k: v for k, v in given.items() if v is not None}}
    env = {**_dotenv(pathlib.Path(".env")), **os.environ}
    base = options.pop("url", None) or env.get("EYEPOP_PIPELINE_URL")
    if not base:
        raise ValueError("pass url or set EYEPOP_PIPELINE_URL")
    if "access_token" not in options and "api_key" not in options:
        if key := env.get("EYEPOP_API_KEY"):
            options["api_key"] = key
        elif token := env.get("EYEPOP_ACCESS_TOKEN"):
            options["access_token"] = token
    return EyePop(base, **options)


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

