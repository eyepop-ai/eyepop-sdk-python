import asyncio
import base64
import contextlib
import inspect
import itertools
import json
import logging
import mimetypes
import os
import pathlib
import struct
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, BinaryIO

import aiohttp

from eyepop.pipeline.errors import EyePopError
from eyepop.pipeline.events import STOP, Done, Dropped, End, Error, Event, Started
from eyepop.pipeline.pop import Ability, Model, Pop
from eyepop.pipeline.prediction import Prediction, Result

if TYPE_CHECKING:
    from eyepop.pipeline.client import EyePop

log = logging.getLogger("eyepop.pipeline")

_CHUNK = 64 * 1024
_CHUNK_KIND = b"\x01"
_COMBINED_KIND = b"\x02"
# shortcut: below this a source goes in one message, above it chunked under the server's credit; the server's
# image-max-bytes is 32 MiB by default, so this stays clear of it; read it from the server if a client needs more
_COMBINED_MAX = 8 * 1024 * 1024

Media = bytes | pathlib.Path | BinaryIO | AsyncIterable[bytes]
Image = str | Mapping[str, Any] | Media
PopLike = Ability | Model | Pop | Mapping[str, Any]


@dataclass(frozen=True)
class _Closed:
    failure: EyePopError | None


@dataclass
class _Upload:
    media: Media
    credit: int = 0
    more: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    uploaded: bool = False


class Source:
    """[EXPERIMENTAL] One source set on a pipeline. Iterate it for its predictions.

    The iteration ends when the source does, and raises `EyePopError` if it failed or the socket did.
    `dropped` counts the predictions the server dropped for a live source because the client was slow.
    """

    def __init__(self, id: str, channel: "asyncio.Queue[Event | _Closed]"):
        self.id = id
        self.dropped = 0
        self._channel = channel
        self._over = False

    async def __aiter__(self) -> AsyncGenerator[Prediction]:
        """Yield each prediction."""
        while not self._over:
            item = await self._channel.get()
            if isinstance(item, Prediction):
                yield item
            elif isinstance(item, Dropped):
                self.dropped += item.count
            elif isinstance(item, End):
                self._over = True
                if item.reason == "error":
                    raise EyePopError(item.message or "the source failed")
            elif isinstance(item, Error):
                self._over = True
                raise EyePopError(item.message, error=item.error)
            elif isinstance(item, _Closed):
                self._over = True
                if item.failure is not None:
                    raise item.failure


class Pipeline:
    """[EXPERIMENTAL] One socket: one pipeline you keep, change the pop of, and set sources on.

    Use it as `async with`, or give it handlers and `run()` it. Nothing opens until then, so
    `source()` may be called first. The pipeline starts before the socket opens, which can take
    seconds on a replica that is short of a GPU.

        async with pipeline(Ability("eyepop.person")) as p:
            async for frame in p.stream("rtsp://camera/1"):
                if frame.objects:
                    await p.set_pop(Ability("eyepop.person").crop(Ability("eyepop.person.2d-body-points")))

    A source's predictions come from its own iterator; `events()` is the tap of everything, and
    must be called before the sources it should see.
    """

    def __init__(self, client: "EyePop | None", pop: PopLike | None = None):
        self._client = client
        self._owned: "EyePop | None" = None
        self._pop = pop
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._outbox: asyncio.Queue[Callable[[], Awaitable[None]]] = asyncio.Queue()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._sources: dict[str, asyncio.Queue[Event | _Closed]] = {}
        self._uploads: dict[str, _Upload] = {}
        self._tap: asyncio.Queue[Event | _Closed] | None = None
        self._handlers: list[tuple[type, str | None, Callable[[Any], Any]]] = []
        self._command_ids = itertools.count(1)
        self._source_ids = itertools.count(1)
        self._tasks: list[asyncio.Task[None]] = []
        self._expected: set[str] | None = None
        self._writing = False
        self._finished = False
        self._closing = False
        self._failure: EyePopError | None = None

    # Opening and closing

    async def __aenter__(self) -> "Pipeline":
        """Open the socket, wait for its pipeline, set the pop."""
        if self._ws is not None or self._closing:
            raise RuntimeError("a pipeline opens once")
        if self._client is None:
            from eyepop.pipeline.client import client

            self._client = self._owned = client()
        deadline = time.monotonic() + self._client._wait  # pyright: ignore[reportPrivateUsage]
        try:
            self._ws = await self._client._retrying(self._client._open_socket, deadline)  # pyright: ignore[reportPrivateUsage]
            self._client._pipelines.add(self)  # pyright: ignore[reportPrivateUsage]
            self._tasks.append(asyncio.create_task(self._read()))
            if self._pop is not None:
                await self.set_pop(self._pop)
            self._writing = True
            self._tasks.append(asyncio.create_task(self._write()))
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the socket; the pipeline stops with it."""
        await self.close()

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        tasks = [*self._tasks, *(u.task for u in self._uploads.values() if u.task)]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._finish(self._failure)
        if self._ws is not None:
            await self._ws.close()
        if self._client is not None:
            self._client._pipelines.discard(self)  # pyright: ignore[reportPrivateUsage]
        if self._owned is not None:
            await self._owned.close()

    # Commands

    async def set_pop(self, pop: PopLike) -> None:
        """Change the pipeline's pop. Answered once the pipeline runs it; a running source keeps running."""
        await self._command({"type": "setPop", "pop": _pop_json(pop)})

    async def graph(self, format: str = "dot", verbose: bool = False) -> str | bytes:
        """The pipeline's graph: DOT as `str`, or the `bytes` of a rendered `png` or `pdf`."""
        answer = await self._command({"type": "getGraph", "format": format, "verbose": verbose})
        if "text" in answer:
            return str(answer["text"])
        return base64.b64decode(answer["data"])

    def source(
        self,
        media: Image,
        *,
        id: str | None = None,
        live: bool | None = None,
        mode: str = "queue",
        content_type: str | None = None,
        **members: Any,
    ) -> Source:
        """Set a source; it is sent once the socket is open, and its predictions come from the returned `Source`.

        `media` is a url (a `str` with `://`), a path (any other `str`, or a `pathlib.Path`), bytes, a
        binary file, an async iterable of bytes, or the command's own members as a mapping, such as
        `{"assetUuid": ...}`. `mode` is `queue`, which plays it after the sources set before it, or
        `preempt`, which drops them all. `members` are more members of the command: `params`, `roi`, `fps`.
        """
        channel: asyncio.Queue[Event | _Closed] = asyncio.Queue()
        return Source(self._add(media, id, live, mode, content_type, members, channel), channel)

    def stream(self, url: str, **members: Any) -> Source:
        """A live source: it runs until the caller stops iterating and closes the pipeline."""
        return self.source(url, live=True, **members)

    def video(self, video: str | Media, content_type: str | None = None, **members: Any) -> Source:
        """A video: a url, a path, bytes, a binary file or an async iterable of bytes."""
        return self.source(video, content_type=content_type, **members)

    async def batch(self, images: Iterable[Image], **members: Any) -> AsyncGenerator[Result]:
        """Queue each image and yield its prediction, or its error when it failed, as it arrives."""
        channel: asyncio.Queue[Event | _Closed] = asyncio.Queue()
        sources = {self._add(image, None, None, "queue", None, members, channel): image for image in images}
        pending = len(sources)
        while pending:
            item = await channel.get()
            image = sources.get(getattr(item, "source", None) or "")
            if isinstance(item, Prediction):
                yield Result(image, item)
            elif isinstance(item, End | Error):
                pending -= 1
                if isinstance(item, Error) or item.reason == "error":
                    yield Result(image, error=item.message or "the item failed")
            elif isinstance(item, _Closed):
                if item.failure is not None:
                    raise item.failure
                return

    # Events and handlers

    def events(self) -> AsyncIterator[Event]:
        """Every event of every source, in order, until the socket closes. Raises `EyePopError` if it failed."""
        queue: asyncio.Queue[Event | _Closed] = asyncio.Queue()
        self._tap = queue
        if self._finished:
            queue.put_nowait(_Closed(self._failure))
        return self._drain(queue)

    def on_prediction(self, handler: Callable[[Prediction], Any] | None = None, *, source: str | None = None) -> Any:
        """Register a handler of each prediction, of one source if `source` is given; also `@pipe.on_prediction(source="cam")`."""
        return self._on(Prediction, handler, source)

    def on_start(self, handler: Callable[[Started], Any] | None = None, *, source: str | None = None) -> Any:
        """Register a handler of each source the pipeline starts to play."""
        return self._on(Started, handler, source)

    def on_end(self, handler: Callable[[End], Any] | None = None, *, source: str | None = None) -> Any:
        """Register a handler of each source that ends, with its reason."""
        return self._on(End, handler, source)

    def on_dropped(self, handler: Callable[[Dropped], Any] | None = None, *, source: str | None = None) -> Any:
        """Register a handler of predictions the server dropped for a slow client."""
        return self._on(Dropped, handler, source)

    def on_error(self, handler: Callable[[Error], Any] | None = None, *, source: str | None = None) -> Any:
        """Register a handler of a failed command; the socket stays open."""
        return self._on(Error, handler, source)

    def on_done(self, handler: Callable[[Done], Any] | None = None) -> Any:
        """Register a handler of the moment no source is open."""
        return self._on(Done, handler, None)

    async def arun(self) -> None:
        """Open, set the sources added so far, and call the handlers until every source has ended, a handler returns `STOP`, or the socket ends.

        A source a handler adds with `source()` is waited for too. `Done` does not end it: it only says that
        nothing was open at that moment, which two sources queued back to back can both see.
        """
        if not self._sources:
            raise ValueError("add a source with source() before run()")
        expected = self._expected = set(self._sources)
        events = self.events()
        async with self:
            async for event in events:
                for kind, source, handler in list(self._handlers):
                    if isinstance(event, kind) and (source is None or getattr(event, "source", None) == source):
                        result = handler(event)
                        if inspect.isawaitable(result):
                            result = await result
                        if result is STOP:
                            return
                if isinstance(event, End | Error) and event.source is not None:
                    expected.discard(event.source)
                    if not expected:
                        return

    def run(self) -> None:
        """`arun()` for a program with no event loop of its own; blocks until it ends or Ctrl-C."""
        asyncio.run(self.arun())

    # Internals

    def _on(self, kind: type, handler: Callable[[Any], Any] | None, source: str | None) -> Any:
        def register(h: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self._handlers.append((kind, source, h))
            return h

        return register if handler is None else register(handler)

    async def _drain(self, queue: "asyncio.Queue[Event | _Closed]") -> AsyncIterator[Event]:
        while not isinstance(item := await queue.get(), _Closed):
            yield item
        if item.failure is not None:
            raise item.failure

    async def _command(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise self._failure or EyePopError("the pipeline is closed")
        id = f"c{next(self._command_ids)}"
        answer: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[id] = answer
        message = {**body, "id": id}
        if self._ws is not None and not self._writing:
            await self._ws_send(message)  # the first pop, before the writer sends the sources set ahead of it
        else:
            self._outbox.put_nowait(lambda: self._ws_send(message))
        return await answer

    async def _ws_send(self, message: dict[str, Any]) -> None:
        assert self._ws is not None
        await self._ws.send_json(message)

    def _add(
        self,
        media: Image,
        id: str | None,
        live: bool | None,
        mode: str,
        content_type: str | None,
        members: dict[str, Any],
        channel: "asyncio.Queue[Event | _Closed]",
    ) -> str:
        if self._finished:
            raise self._failure or EyePopError("the pipeline is closed")
        id = id or f"s{next(self._source_ids)}"
        if id in self._sources:
            raise ValueError(f"source id {id!r} is open already")
        message: dict[str, Any] = {"type": "setSource", "id": id, **members}
        if live is not None:
            message["live"] = live
        if mode != "queue":
            message["mode"] = mode
        if isinstance(media, str):
            media = {"url": media} if "://" in media else pathlib.Path(media)
        upload: Media | None = None
        if isinstance(media, Mapping):
            message.update(media)
        else:
            upload = media
            name = None if isinstance(media, bytes) else getattr(media, "name", None)
            message["upload"] = {"contentType": _content_type(name, content_type)}
        self._sources[id] = channel
        if self._expected is not None:
            self._expected.add(id)
        self._outbox.put_nowait(lambda: self._send_source(message, upload))
        return id

    async def _send_source(self, message: dict[str, Any], media: Media | None) -> None:
        assert self._ws is not None
        if media is None:
            await self._ws.send_json(message)
        elif (size := _size(media)) is not None and size <= _COMBINED_MAX:
            header = json.dumps(message).encode()
            body = await _read_all(media)
            await self._ws.send_bytes(_COMBINED_KIND + struct.pack(">I", len(header)) + header + body)
        else:
            self._uploads[message["id"]] = _Upload(media)  # before the command, so that `send` finds it
            await self._ws.send_json(message)

    async def _write(self) -> None:
        try:
            while True:
                await (await self._outbox.get())()
        except Exception as e:
            self._abort(EyePopError(f"sending to the pipeline API failed: {type(e).__name__}: {e}"), e)

    async def _read(self) -> None:
        assert self._ws is not None
        try:
            while (msg := await self._ws.receive()).type == aiohttp.WSMsgType.TEXT:
                self._dispatch(msg.data)
        except EyePopError as e:
            self._abort(e)
            return
        self._finish(self._failure or _socket_failure(msg, self._closing))

    def _dispatch(self, data: str) -> None:
        try:
            message = json.loads(data)
        except ValueError:
            message = None
        if not isinstance(message, dict):
            raise EyePopError("the pipeline API sent a message that is not a JSON object")
        kind, id = message.get("type"), message.get("id")
        if not isinstance(id, str):
            id = ""  # a message of no source or command, such as `done`
        if kind == "result":
            if (answer := self._pending.pop(id, None)) and not answer.done():
                answer.set_result(message)
        elif kind == "send" and (upload := self._uploads.get(id)) and upload.task is None:
            upload.credit = int(message.get("credit", 0))
            upload.task = asyncio.create_task(self._pump(id, upload))
        elif kind == "credit" and (upload := self._uploads.get(id)):
            upload.credit += int(message.get("bytes", 0))
            upload.more.set()
        elif kind == "error":
            self._fail_command(message)
        elif kind == "start":
            self._deliver(Started(id), id)
        elif kind == "prediction":
            try:
                self._deliver(replace(Prediction.parse(_mapping(message.get("prediction"))), source=id), id)
            except (AttributeError, TypeError, ValueError) as e:
                raise EyePopError("the pipeline API sent a prediction that is not one") from e
        elif kind == "dropped":
            self._deliver(Dropped(id, int(message.get("count", 0))), id)
        elif kind == "end":
            self._deliver(End(id, message.get("reason", ""), message.get("message", "")), id)
            self._source_over(id)
        elif kind == "done":
            self._deliver(Done(), "")

    def _fail_command(self, message: dict[str, Any]) -> None:
        id, code, text = message.get("id") or "", message.get("error", ""), message.get("message", "")
        if (answer := self._pending.pop(id, None)) is not None:
            if not answer.done():
                answer.set_exception(EyePopError(text, error=code))
            return
        self._deliver(Error(id or None, code, text), id)
        self._source_over(id)

    def _deliver(self, event: Event, source: str) -> None:
        if (channel := self._sources.get(source)) is not None:
            channel.put_nowait(event)
        if self._tap is not None:
            self._tap.put_nowait(event)

    def _source_over(self, id: str) -> None:
        self._sources.pop(id, None)
        upload = self._uploads.pop(id, None)
        if upload is not None and upload.task is not None:
            # the bytes still on their way are discarded by the server; `uploaded` lets the next upload start
            upload.task.cancel()
            self._outbox.put_nowait(lambda: self._send_uploaded(id, upload))

    async def _pump(self, id: str, upload: _Upload) -> None:
        try:
            async for chunk in _chunks(upload.media):
                view = memoryview(chunk)
                while view:
                    while upload.credit == 0:
                        upload.more.clear()
                        await upload.more.wait()
                    n = min(len(view), upload.credit)
                    assert self._ws is not None
                    await self._ws.send_bytes(_CHUNK_KIND + bytes(view[:n]))
                    upload.credit -= n
                    view = view[n:]
            await self._send_uploaded(id, upload)
        except Exception as e:
            self._abort(EyePopError(f"sending the media of source {id!r} failed"), e)

    async def _send_uploaded(self, id: str, upload: _Upload) -> None:
        if not upload.uploaded:
            upload.uploaded = True
            await self._ws_send({"type": "uploaded", "id": id})

    def _abort(self, failure: EyePopError, cause: BaseException | None = None) -> None:
        if cause is not None:
            failure.__cause__ = cause
        self._failure = self._failure or failure
        self._finish(self._failure)
        if self._ws is not None and not self._ws.closed:
            asyncio.ensure_future(self._ws.close())

    def _finish(self, failure: EyePopError | None) -> None:
        if self._finished:
            return
        self._finished = True
        self._failure = failure
        for answer in self._pending.values():
            if not answer.done():
                answer.set_exception(failure or EyePopError("the pipeline is closed"))
        self._pending.clear()
        closed = _Closed(failure)
        for channel in {*self._sources.values(), *([self._tap] if self._tap else [])}:
            channel.put_nowait(closed)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("not an object")
    return value


def _pop_json(pop: PopLike) -> Mapping[str, Any]:
    return pop if isinstance(pop, Mapping) else pop.json


def _socket_failure(msg: aiohttp.WSMessage, closed_by_client: bool) -> EyePopError | None:
    if msg.type == aiohttp.WSMsgType.CLOSE:
        code, reason = msg.data, msg.extra or None
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


def _content_type(name: object, content_type: str | None) -> str:
    # a file opened from a descriptor is named by an int
    guessed = mimetypes.guess_type(name)[0] if isinstance(name, str) else None
    return content_type or guessed or "application/octet-stream"


def _size(media: Media) -> int | None:
    if isinstance(media, bytes):
        return len(media)
    if isinstance(media, pathlib.Path):
        return media.stat().st_size
    if isinstance(media, AsyncIterable):
        return None
    try:
        position = media.tell()
        end = media.seek(0, os.SEEK_END)
        media.seek(position)
        return end - position
    except (OSError, ValueError, AttributeError):
        return None


async def _read_all(media: Media) -> bytes:
    if isinstance(media, bytes):
        return media
    if isinstance(media, pathlib.Path):
        return await asyncio.to_thread(media.read_bytes)
    assert not isinstance(media, AsyncIterable)
    return await asyncio.to_thread(media.read)


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
