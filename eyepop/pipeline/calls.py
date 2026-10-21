import asyncio
import contextlib
import inspect
from collections.abc import AsyncGenerator, Callable, Iterable
from typing import Any

from eyepop.pipeline.client import client
from eyepop.pipeline.events import STOP
from eyepop.pipeline.prediction import Prediction, Result
from eyepop.pipeline.session import Image, Media, Pipeline, PopLike


async def image(pop: PopLike, image: Image, content_type: str | None = None) -> Prediction:
    """[EXPERIMENTAL] One image's prediction, over one `POST`. It exchanges the API key each call: to send many, hold `client()`."""
    async with client() as ep:
        return await ep.use(pop).image(image, content_type)


def pipeline(pop: PopLike | None = None) -> Pipeline:
    """[EXPERIMENTAL] A pipeline to keep, on a client of its own that it closes with itself."""
    return Pipeline(None, pop)


class Job[T]:
    """What `batch`, `video` and `stream` return: iterate it, or decorate a handler with it.

        async for frame in stream(person, url): ...

        @stream(person, url)
        async def watch(frame): ...

        await watch()   # or watch.run()
    """

    def __init__(self, pop: PopLike, method: str, *args: Any):
        self._pop = pop
        self._method = method
        self._args = args

    def __aiter__(self) -> AsyncGenerator[T]:
        """Open a pipeline of its own and yield from it."""
        return self._items()

    def __call__(self, handler: Callable[[T], Any]) -> "Handled[T]":
        """Hand each item to `handler`."""
        return Handled(self, handler)

    async def _items(self) -> AsyncGenerator[T]:
        async with client() as ep:
            async for item in getattr(ep.use(self._pop), self._method)(*self._args):
                yield item


class Handled[T]:
    """A job with its handler. Call it to run it; it ends with the media, or when the handler returns `STOP`."""

    def __init__(self, job: Job[T], handler: Callable[[T], Any]):
        self._job = job
        self._handler = handler

    async def __call__(self) -> None:
        """Run it."""
        async with contextlib.aclosing(self._job.__aiter__()) as items:
            async for item in items:
                result = self._handler(item)
                if inspect.isawaitable(result):
                    result = await result
                if result is STOP:
                    return

    def run(self) -> None:
        """Run it in a program with no event loop of its own."""
        asyncio.run(self())


def batch(pop: PopLike, images: Iterable[Image]) -> Job[Result]:
    """[EXPERIMENTAL] Each image's prediction, or its error, as it arrives over one socket."""
    return Job(pop, "batch", images)


def video(pop: PopLike, video: str | Media, content_type: str | None = None) -> Job[Prediction]:
    """[EXPERIMENTAL] Each frame's prediction of a video: a url, a path, bytes, a binary file or an async iterable of bytes."""
    return Job(pop, "video", video, content_type)


def stream(pop: PopLike, url: str) -> Job[Prediction]:
    """[EXPERIMENTAL] Each frame's prediction of a live stream, until you stop iterating or the handler returns `STOP`."""
    return Job(pop, "stream", url)


async def run(*handled: Handled[Any]) -> None:
    """[EXPERIMENTAL] Run decorated jobs together, each on a pipeline of its own."""
    await asyncio.gather(*(h() for h in handled))
