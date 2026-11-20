"""Reading a recording from the media cache by its mediaId (eyepop-instance spec 025).

A source that records to the media cache (`media_cache_seconds`) gets a mediaId: in pipeline mode from
its `start`, on a session from the `recording` event that opens its result stream. Every read is a
REST request: a session's endpoint asks its instance (`GET {instance}/media/{mediaId}/...`), a pipeline
endpoint the pipeline API (`GET {pipeline API}/api/v1/media/{mediaId}/...`). Both answer alike.
"""
import asyncio
import json
import logging
import math
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from eyepop.exceptions import MediaReadException
from eyepop.settings import settings

if TYPE_CHECKING:
    from eyepop.worker.worker_backend import WorkerBackend

log_requests = logging.getLogger('eyepop.requests')

MediaRead = Literal['range', 'frame', 'clip']
ClipMode = Literal['exact', 'keyframe']

# A read waits on the server for up to its `wait`; a clip then streams for as long as it is read.
MEDIA_READ_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)

_CLIP_CHUNK_BYTES = 64 * 1024


class MediaClip:
    """A clip of a recording, an MP4 that streams: iterate it for its bytes, or `read()` it whole.

    `start_ns` and `end_ns` are the clip's real bounds, from `X-Start-Ns` and `X-End-Ns`; a keyframe clip
    starts at the keyframe at or before the asked start. `milli_compute_units` is what the pipeline API
    priced the read at, from `X-Milli-Compute-Units`, and None when the answer carries no such header
    (an instance never prices a read). The clip holds a connection until it is read to its end or
    closed, so read it, iterate it to its end, or use it as an async context manager.
    """

    def __init__(self, response: aiohttp.ClientResponse):
        self._response = response
        self.start_ns: int | None = _int_header(response.headers, 'X-Start-Ns')
        self.end_ns: int | None = _int_header(response.headers, 'X-End-Ns')
        self.milli_compute_units: int | None = _int_header(response.headers, 'X-Milli-Compute-Units')
        self.content_type: str = response.headers.get('Content-Type', 'video/mp4')

    def __aiter__(self) -> AsyncIterator[bytes]:
        """The clip's bytes, chunk by chunk; the connection is let go at the end."""
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.content.iter_chunked(_CLIP_CHUNK_BYTES):
                yield chunk
        finally:
            self._response.release()

    async def read(self) -> bytes:
        """The whole clip."""
        try:
            return await self._response.read()
        finally:
            self._response.release()

    def close(self) -> None:
        """Let the clip go before its end; the server stops encoding it."""
        self._response.close()

    async def __aenter__(self) -> 'MediaClip':
        """The clip, closed when the block ends."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the clip."""
        self.close()


async def read_range(backend: 'WorkerBackend', media_id: str) -> tuple[int, int]:
    """The first and last nanosecond the media cache holds of the recording."""
    response = await _request(backend, media_id, 'range', {}, accept='application/json')
    try:
        body = await response.json()
    finally:
        response.release()
    return int(body['start_ns']), int(body['end_ns'])


async def read_frame(
        backend: 'WorkerBackend',
        media_id: str,
        timestamp_ns: int,
        *,
        bbox: Sequence[float] | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
        png: bool = False,
        wait_ms: int | None = None,
) -> bytes:
    """The frame at a prediction's timestamp, JPEG or with `png` PNG; see WorkerEndpoint.media_frame."""
    params: dict[str, Any] = {'timestamp': _nanoseconds(timestamp_ns, 'timestamp_ns')}
    if bbox is not None:
        params['bbox'] = bbox_param(bbox)
    if max_width is not None:
        params['maxWidth'] = max_width
    if max_height is not None:
        params['maxHeight'] = max_height
    if wait_ms is not None:
        params['wait'] = wait_ms
    response = await _request(backend, media_id, 'frame', params, accept='image/png' if png else 'image/jpeg')
    try:
        return await response.read()
    finally:
        response.release()


async def open_clip(
        backend: 'WorkerBackend',
        media_id: str,
        start_ns: int,
        end_ns: int,
        *,
        mode: ClipMode = 'exact',
        wait_ms: int | None = None,
) -> MediaClip:
    """A clip between two timestamps; see WorkerEndpoint.media_clip."""
    params: dict[str, Any] = {
        'start': _nanoseconds(start_ns, 'start_ns'),
        'end': _nanoseconds(end_ns, 'end_ns'),
        'mode': mode,
    }
    if wait_ms is not None:
        params['wait'] = wait_ms
    return MediaClip(await _request(backend, media_id, 'clip', params, accept='video/mp4'))


def bbox_param(bbox: Sequence[float]) -> str:
    """`x,y,w,h` in whole pixels: the smallest box of whole pixels that holds the given one."""
    if len(bbox) != 4:
        raise ValueError(f'a bbox is (x, y, width, height), not {bbox!r}')
    x, y, width, height = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (x, y, width, height)):
        raise ValueError(f'a bbox has finite coordinates, not {bbox!r}')
    left, top = math.floor(x), math.floor(y)
    right, bottom = math.ceil(x + width), math.ceil(y + height)
    return f'{left},{top},{right - left},{bottom - top}'


def frame_box(obj: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """An object's box, `x`, `y`, `width` and `height` in the source's pixels, which are the recording's."""
    try:
        return float(obj['x']), float(obj['y']), float(obj['width']), float(obj['height'])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f'the object has no box (x, y, width, height): {e}') from None


def _nanoseconds(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} is a timestamp in nanoseconds, not {value!r}')
    return int(value)


async def _request(backend: 'WorkerBackend', media_id: str, read: MediaRead, params: dict[str, Any],
                   accept: str) -> aiohttp.ClientResponse:
    """The read's answer when it succeeded.

    A `425` (not cached yet) and a `503 overloaded` (the pipeline API's limiter, which reads pass as sockets
    do) are asked again after their Retry-After, within max_retry_time_secs. Each retry, its answer and the
    reading of a refusal's body, is bounded by what is left of that budget; when it runs out the last
    refusal is raised. An accepted clip streams outside the budget: only its headers are waited for here.
    """
    if not media_id:
        raise ValueError('a media read needs a mediaId')
    deadline = time.monotonic() + settings.max_retry_time_secs
    outcome = await _attempt(backend, media_id, read, params, accept)
    while True:
        if isinstance(outcome, aiohttp.ClientResponse):
            return outcome
        error = outcome
        if not _retried(error):
            raise error
        wait = error.retry_after if error.retry_after is not None else 1.0
        if time.monotonic() + wait > deadline:
            raise error
        log_requests.debug('media %s %s refused %d %s, retrying in %.1f s', read, media_id, error.status,
                           error.reason or error.error, wait)
        await asyncio.sleep(wait)
        try:
            outcome = await asyncio.wait_for(_attempt(backend, media_id, read, params, accept),
                                             max(deadline - time.monotonic(), 0.0))
        except TimeoutError:
            raise error from None


async def _attempt(backend: 'WorkerBackend', media_id: str, read: MediaRead, params: dict[str, Any],
                   accept: str) -> aiohttp.ClientResponse | MediaReadException:
    """One read: its answer when it succeeded, else the refusal, its body read and the answer let go."""
    response = await backend.media_get(media_id, read, params, accept)
    if response.ok:
        return response
    return await _error_of(response)


def _retried(error: MediaReadException) -> bool:
    """A moment not cached yet (`425`), or a pipeline API at capacity (`503 overloaded`)."""
    return error.status == 425 or (error.status == 503 and 'overloaded' in (error.error, error.reason))


async def _error_of(response: aiohttp.ClientResponse) -> MediaReadException:
    try:
        text = await response.text()
    except (aiohttp.ClientError, UnicodeDecodeError):
        text = ''
    finally:
        response.release()
    error = reason = None
    message = text or (response.reason or '')
    try:
        body = json.loads(text) if text else None
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = _str_or_none(body.get('error'))
        reason = _str_or_none(body.get('reason'))
        message = _str_or_none(body.get('message')) or message
    return MediaReadException(response.status, error=error, reason=reason, message=message,
                              retry_after=_retry_after(response.headers))


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    try:
        wait = float(headers['Retry-After'])
    except (KeyError, ValueError):
        return None
    # A floor, so that `Retry-After: 0` cannot turn a wait into a request storm.
    return max(wait, 0.5) if math.isfinite(wait) and wait >= 0 else None


def _int_header(headers: Mapping[str, str], name: str) -> int | None:
    try:
        return int(headers[name])
    except (KeyError, ValueError):
        return None
