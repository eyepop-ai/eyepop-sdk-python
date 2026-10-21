"""A WorkerEndpoint backed by the pipeline API's socket (SDK-24).

The socket protocol is eyepop-instance spec 019, `contracts/socket.md`: one
socket runs its own pipeline and takes commands, `setPop` and `setSource`
among them, and streams every source's predictions back under the source's id.
"""
import asyncio
import ipaddress
import json
import logging
import math
import os
import struct
import time
import warnings
from collections.abc import AsyncIterable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlsplit

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.api import refresh_compute_token
from eyepop.data.types.asset import Area
from eyepop.exceptions import (
    PipelineCommandException,
    PipelineSocketClosedException,
    PipelineSourceException,
    PopConfigurationException,
)
from eyepop.jobs import JobStateCallback
from eyepop.settings import settings
from eyepop.worker.camera import Camera
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_jobs import WorkerJob, _guess_mime_type_from_location
from eyepop.worker.worker_types import ComponentParams, MotionDetectConfig, Pop, VideoMode

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')

PIPELINE_PATH = '/api/v1/pipeline'
SUBPROTOCOL = 'eyepop.v1'
# A socket may have this many open sources; one more is answered queue_full.
MAX_OPEN_SOURCES = 256

_CHUNK = 0x01
_COMBINED = 0x02
_CHUNK_BYTES = 64 * 1024
_REFRESH_BEFORE_EXPIRY_SECS = 60.0
# The closes after which the endpoint opens a new socket at once; after any
# other close it opens one when the next job or set_pop needs it.
_RECONNECT_AT_ONCE = {(1008, 'credential_expired'), (1011, 'overloaded'), (1001, 'shutting_down')}
# What the socket's setSource does not take yet (AWSU-317).
_NOT_SENT = ('video_mode', 'motion_detect', 'media_cache_seconds')


@dataclass
class _Media:
    """One upload's bytes: an opener for a stream, its content type, and its size when it is a file the job opens.

    The job closes a file it opened; a caller's stream stays open.
    """
    open: Callable[[], BinaryIO | AsyncIterable[bytes]]
    content_type: str
    size: int | None = None

    @property
    def owned(self) -> bool:
        return self.size is not None


class _UploadEnded(Exception):
    pass


class _Upload:
    """A chunked upload's credit, granted by the server's `send` and `credit`."""

    def __init__(self):
        self.credit = 0
        self.granted = asyncio.Event()
        self.ended = False

    def grant(self, n: int):
        self.credit += n
        self.granted.set()

    def end(self):
        self.ended = True
        self.granted.set()

    async def take(self, most: int) -> int:
        while self.credit <= 0 and not self.ended:
            self.granted.clear()
            await self.granted.wait()
        if self.ended:
            raise _UploadEnded()
        n = min(most, self.credit)
        self.credit -= n
        return n


_CANCELLED = object()


@dataclass
class _Source:
    """A job's source on one socket: the messages for its id, and its chunked uploads by member."""
    id: str
    inbox: asyncio.Queue = field(default_factory=asyncio.Queue)
    uploads: dict[int | None, _Upload] = field(default_factory=dict)

    def close(self, error: BaseException):
        for upload in self.uploads.values():
            upload.end()
        self.inbox.put_nowait(error)


class _Socket:
    """One upgraded socket: its open sources and commands by id, and the task that reads it."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse):
        self.ws = ws
        self.sources: dict[str, _Source] = {}
        self.commands: dict[str, asyncio.Future] = {}
        self.closed = asyncio.Event()
        self.reader: asyncio.Task | None = None
        self.write_lock = asyncio.Lock()

    @property
    def open(self) -> bool:
        return not self.closed.is_set() and not self.ws.closed

    async def send_json(self, message: dict[str, Any]):
        async with self.write_lock:
            await self.ws.send_str(json.dumps(message))

    async def send_bytes(self, data: bytes):
        async with self.write_lock:
            await self.ws.send_bytes(data)


class _SocketJob(WorkerJob):
    """A job run as one source on the endpoint's socket.

    `source` holds the setSource members that name the source (`url`,
    `assetUuid`, `upload` or `group`); `media` the bytes of its upload members,
    in member order.
    """

    def __init__(
            self,
            session: 'PipelineEndpoint',
            source: dict[str, Any],
            media: list[_Media],
            live: bool,
            component_params: list[ComponentParams] | None,
            roi: Area | None,
            fps: str | None,
            camera: Camera | None,
            on_ready: Callable[[WorkerJob], None] | None,
            callback: JobStateCallback | None,
    ):
        super().__init__(
            session=session,
            component_params=component_params,
            motion_detect=None,
            roi=roi,
            fps=fps,
            camera=camera,
            media_cache_seconds=None,
            on_ready=on_ready,
            callback=callback,
        )
        self.source = source
        self.media = media
        self.live = live
        self.source_on_socket: _Source | None = None

    def command(self, id: str) -> dict[str, Any]:
        command: dict[str, Any] = {'type': 'setSource', 'id': id, **self.source}
        if self._component_params is not None:
            command['params'] = TypeAdapter(list[ComponentParams]).dump_python(self._component_params)
        if self._roi is not None:
            command['roi'] = self._roi.model_dump(exclude_none=True)
        if self._camera is not None:
            command['camera'] = self._camera.model_dump(exclude_none=True)
        if self._fps is not None:
            command['fps'] = self._fps
        if self.live:
            command['live'] = True
        return command

    @property
    def cancelled(self) -> bool:
        return self._queue is None

    async def cancel(self):
        await super().cancel()
        if self.source_on_socket is not None:
            self.source_on_socket.inbox.put_nowait(_CANCELLED)

    async def _do_execute_job(self, queue: asyncio.Queue, session: Any):
        await session._run(self)


class PipelineEndpoint(WorkerEndpoint):
    """A WorkerEndpoint that runs its jobs on the pipeline API's socket, selected by `is_pipeline_mode`.

    One socket per endpoint, opened at connect(). Jobs go out in the order
    they were submitted, each as one source queued behind the ones before it.
    """

    def __init__(
            self,
            access_token: str | None,
            api_key: str | None,
            eyepop_url: str,
            job_queue_length: int,
            pop: Pop | dict[str, Any] | None = None,
            is_local_mode: bool = False,
            account_id: str | None = None,
    ):
        if job_queue_length > MAX_OPEN_SOURCES:
            raise ValueError(
                f'job_queue_length {job_queue_length} exceeds {MAX_OPEN_SOURCES}, '
                f'the open sources a pipeline API socket takes')
        super().__init__(
            secret_key=None,
            access_token=access_token,
            api_key=api_key,
            eyepop_url=eyepop_url,
            pop_id='transient',
            session_uuid=None,
            auto_start=True,
            stop_jobs=False,
            job_queue_length=job_queue_length,
            # The request tracer posts to a worker's /events, which neither the
            # compute API nor the pipeline API serves.
            request_tracer_max_buffer=0,
            pop=pop,
            is_local_mode=is_local_mode,
            account_id=account_id,
        )
        self.pipeline_api_url: str | None = None
        self._socket: _Socket | None = None
        self._send_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._reconnecting: asyncio.Task | None = None
        self._disconnecting = False
        self._token: str | None = None
        self._token_expires = 0.0
        self._next_id = 0
        self._warned: set[str] = set()

    # Connection

    async def _reconnect(self):
        async with self._send_lock:
            await self._ensure_socket()

    async def _disconnect(self, timeout: float | None = None):
        self._disconnecting = True
        tasks = list(self.tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._reconnecting is not None:
            self._reconnecting.cancel()
            await asyncio.gather(self._reconnecting, return_exceptions=True)
            self._reconnecting = None
        socket, self._socket = self._socket, None
        if socket is not None:
            await socket.ws.close()
            if socket.reader is not None:
                await asyncio.gather(socket.reader, return_exceptions=True)

    async def connect(self):
        self._disconnecting = False
        await super().connect()

    async def _ensure_socket(self) -> _Socket:
        """The open socket, or a new one with the last pop set on it. The caller holds the send lock."""
        async with self._connect_lock:
            if self._socket is not None and self._socket.open:
                return self._socket
            if self.client_session is None:
                raise PipelineSocketClosedException(None, 'the endpoint is not connected')
            url = await self._socket_url()
            ws = await self._upgrade(url)
            socket = _Socket(ws)
            socket.reader = asyncio.create_task(self._read(socket))
            self._socket = socket
            if self.pop is not None:
                try:
                    await self._set_pop_on(socket, self.pop)
                except PopConfigurationException:
                    # A socket whose pipeline does not run the pop runs no job.
                    self._socket = None
                    await socket.ws.close()
                    raise
            return socket

    async def _socket_url(self) -> str:
        if self.pipeline_api_url is None:
            if self.is_local_mode:
                base = self.eyepop_url
            else:
                base = urljoin(self.eyepop_url, await self._configured_pipeline_api_url())
            self.pipeline_api_url = _checked_url(base)
        return 'ws' + self.pipeline_api_url.removeprefix('http') + PIPELINE_PATH

    async def _configured_pipeline_api_url(self) -> str:
        assert self.client_session is not None
        config_url = f'{self.eyepop_url.rstrip("/")}/v1/configs'
        if self.compute_ctx is not None and self.compute_ctx.account_uuid:
            config_url = f'{config_url}?account_uuid={self.compute_ctx.account_uuid}'
        headers = {}
        token = await self._pipeline_token()
        if token is not None:
            headers['Authorization'] = f'Bearer {token}'
        async with self.client_session.get(config_url, headers=headers) as response:
            config = await response.json()
        url = config.get('pipeline_api_url') if isinstance(config, dict) else None
        if not url:
            raise PopConfigurationException(
                'transient', f'{config_url} has no pipeline_api_url: this compute API serves no pipeline API')
        return url

    async def _pipeline_token(self, force: bool = False) -> str | None:
        """The user access token the socket sends; an API key is exchanged at the compute API and never sent."""
        if self.provided_access_token is not None:
            return self.provided_access_token
        if self.compute_ctx is None or not self.compute_ctx.api_key:
            return None
        if force or self._token is None or self._token_expires - time.monotonic() < _REFRESH_BEFORE_EXPIRY_SECS:
            assert self.client_session is not None
            self.compute_ctx = await refresh_compute_token(self.compute_ctx, self.client_session)
            self._token = self.compute_ctx.m2m_access_token
            self._token_expires = time.monotonic() + float(self.compute_ctx.access_token_expires_in or 0)
        return self._token

    async def _upgrade(self, url: str) -> aiohttp.ClientWebSocketResponse:
        """Open the socket, waiting through 503 with Retry-After and exchanging the API key again after a 401."""
        assert self.client_session is not None
        deadline = time.monotonic() + settings.max_retry_time_secs
        rejected = False
        while True:
            token = await self._pipeline_token(force=rejected)
            headers = {'Authorization': f'Bearer {token}'} if token is not None else {}
            try:
                log_requests.debug('before upgrade %s', url)
                return await self.client_session.ws_connect(
                    url,
                    protocols=(SUBPROTOCOL,),
                    headers=headers,
                    heartbeat=settings.pipeline_heartbeat_secs,
                    max_msg_size=0,
                )
            except aiohttp.ClientResponseError as e:
                if e.status == 401 and not rejected and self.provided_access_token is None and self._token is not None:
                    log_requests.debug('the pipeline API answered 401, exchanging the API key again')
                    rejected = True
                    continue
                wait = _retry_after(e.status, e.headers)
                if wait is None or time.monotonic() + wait > deadline:
                    raise
                log_requests.info('the pipeline API answered %d, retrying in %.1f s', e.status, wait)
                await asyncio.sleep(wait)

    async def _read(self, socket: _Socket):
        code: int | None = None
        reason = ''
        try:
            while True:
                message = await socket.ws.receive()
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        self._dispatch(socket, json.loads(message.data))
                    except (ValueError, TypeError, KeyError) as e:
                        log.warning('unreadable message from the pipeline API: %s', e)
                elif message.type == aiohttp.WSMsgType.CLOSE:
                    code, reason = message.data, message.extra or ''
                    break
                elif message.type in (aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            if code is None:
                code = socket.ws.close_code
            self._closed(socket, code, reason)

    def _closed(self, socket: _Socket, code: int | None, reason: str):
        socket.closed.set()
        if self._socket is socket:
            self._socket = None
        error = PipelineSocketClosedException(code, reason)
        for source in socket.sources.values():
            source.close(error)
        for answer in socket.commands.values():
            if not answer.done():
                answer.set_exception(error)
        if self._disconnecting:
            return
        log_requests.debug('the pipeline API closed the socket: %s %s', code, reason)
        if (code, reason) in _RECONNECT_AT_ONCE:
            if reason == 'credential_expired':
                self._token = None
            self._reconnecting = asyncio.create_task(self._reconnect_at_once())

    async def _reconnect_at_once(self):
        try:
            async with self._send_lock:
                await self._ensure_socket()
        except Exception as e:
            # The next job or set_pop tries again and reports what fails.
            log_requests.info('could not reopen the pipeline API socket: %s', e)

    def _dispatch(self, socket: _Socket, message: dict[str, Any]):
        type_, id_ = message.get('type'), message.get('id')
        source = socket.sources.get(id_) if isinstance(id_, str) else None
        if type_ in ('send', 'credit'):
            upload = source.uploads.get(message.get('member')) if source is not None else None
            if upload is not None:
                upload.grant(int(message['credit'] if type_ == 'send' else message['bytes']))
            return
        if isinstance(id_, str) and id_ in socket.commands:
            answer = socket.commands.pop(id_)
            if answer.done():
                return
            if type_ == 'error':
                answer.set_exception(PipelineCommandException(message.get('error', ''), message.get('message', '')))
            else:
                answer.set_result(message)
            return
        if source is not None:
            if type_ == 'end':
                for upload in source.uploads.values():
                    upload.end()
            source.inbox.put_nowait(message)
        elif type_ == 'error':
            log.warning('the pipeline API refused a message: %s %s', message.get('error'), message.get('message'))

    # Pop

    async def get_pop(self) -> Pop | None:
        return self.pop

    async def set_pop(self, pop: Pop):
        """Set the pop on the socket's pipeline; a pop it refuses raises, and the last pop it took stays."""
        if self.client_session is None:
            self.pop = pop
            return {'pop': pop.model_dump()}
        async with self._send_lock:
            if self._socket is not None and self._socket.open:
                await self._set_pop_on(self._socket, pop)
                self.pop = pop
            else:
                previous, self.pop = self.pop, pop
                try:
                    await self._ensure_socket()
                except BaseException:
                    self.pop = previous
                    raise
        return {'pop': pop.model_dump()}

    async def _set_pop_on(self, socket: _Socket, pop: Pop):
        id = self._new_id('pop')
        answer = asyncio.get_running_loop().create_future()
        socket.commands[id] = answer
        await socket.send_json({'type': 'setPop', 'id': id, 'pop': json.loads(pop.model_dump_json())})
        try:
            await answer
        except PipelineCommandException as e:
            raise PopConfigurationException('transient', f'{e.error}: {e.message}') from e

    def _new_id(self, prefix: str) -> str:
        self._next_id += 1
        return f'{prefix}-{self._next_id}'

    # Jobs

    async def _run(self, job: _SocketJob):
        """Run the job's source to its end; a live source refused overloaded goes once more on a new socket."""
        retried = False
        while True:
            socket, source = await self._set_source(job)
            uploader = None
            if source.uploads:
                uploader = asyncio.create_task(self._upload(socket, source, job))
            try:
                overloaded = await self._follow(job, source)
            finally:
                job.source_on_socket = None
                socket.sources.pop(source.id, None)
                if uploader is not None:
                    uploader.cancel()
                    await asyncio.gather(uploader, return_exceptions=True)
            if not overloaded:
                return
            if retried or not job.live:
                raise PipelineCommandException('overloaded', 'the pipeline API has no capacity for another live source')
            retried = True
            await socket.closed.wait()

    async def _set_source(self, job: _SocketJob) -> tuple[_Socket, _Source]:
        """Send the job's setSource, after every job submitted before it.

        The send lock is the first thing a job waits for, and asyncio.Lock wakes
        its waiters in order, so jobs go out in the order they were submitted.
        """
        async with self._send_lock:
            combined = await _combined_media(job)
            socket = await self._ensure_socket()
            source = _Source(self._new_id('source'))
            command = job.command(source.id)
            if combined is None:
                for key in _upload_keys(job):
                    source.uploads[key] = _Upload()
            socket.sources[source.id] = source
            job.source_on_socket = source
            if combined is None:
                await socket.send_json(command)
            else:
                await socket.send_bytes(_combined_message(command, combined))
        return socket, source

    async def _follow(self, job: _SocketJob, source: _Source) -> bool:
        """Push the source's predictions to the job until it ends; True when the pipeline API was overloaded."""
        first = True
        while True:
            if job.cancelled:
                return False
            message = await source.inbox.get()
            if message is _CANCELLED:
                return False
            if isinstance(message, BaseException):
                raise message
            type_ = message.get('type')
            if type_ == 'prediction':
                if first:
                    job._callback.first_result(job)
                    first = False
                await job.push_message(message.get('prediction'))
            elif type_ == 'end':
                reason = message.get('reason')
                if reason == 'eos':
                    return False
                raise PipelineSourceException(source.id, reason or '', message.get('message', ''))
            elif type_ == 'error':
                if message.get('error') == 'overloaded':
                    return True
                raise PipelineCommandException(message.get('error', ''), message.get('message', ''))
            elif type_ == 'dropped':
                log.debug('the pipeline API dropped %s predictions of a live source', message.get('count'))

    async def _upload(self, socket: _Socket, source: _Source, job: _SocketJob):
        """Send each upload member's bytes in chunks within its credit, then its uploaded."""
        try:
            for key, media in zip(_upload_keys(job), job.media, strict=True):
                upload = source.uploads[key]
                stream = media.open()
                try:
                    async for piece in _pieces(stream):
                        while piece:
                            n = await upload.take(min(len(piece), _CHUNK_BYTES))
                            await socket.send_bytes(bytes([_CHUNK]) + piece[:n])
                            piece = piece[n:]
                finally:
                    if media.owned and not isinstance(stream, AsyncIterable):
                        stream.close()
                uploaded: dict[str, Any] = {'type': 'uploaded', 'id': source.id}
                if key is not None:
                    uploaded['member'] = key
                await socket.send_json(uploaded)
        except (_UploadEnded, ConnectionError, aiohttp.ClientError):
            # The source ended or the socket closed; the job reports how.
            pass
        except Exception as e:
            source.inbox.put_nowait(e)

    def _not_sent(self, **options: Any):
        for name, value in options.items():
            if value is None or name in self._warned:
                continue
            self._warned.add(name)
            message = f'{name} is not sent to the pipeline API and has no effect in pipeline mode'
            if name in _NOT_SENT:
                message += ' (AWSU-317)'
            warnings.warn(message, UserWarning, stacklevel=3)

    async def _submit(self, job: _SocketJob) -> WorkerJob:
        await self._task_start(job.execute())
        return job

    def _job(self, source: dict[str, Any], media: list[_Media], live: bool, params: list[ComponentParams] | None,
             roi: Area | None, camera: Camera | None, fps: str | None,
             on_ready: Callable[[WorkerJob], None] | None) -> _SocketJob:
        return _SocketJob(session=self, source=source, media=media, live=live, component_params=params, roi=roi,
                          fps=fps, camera=camera, on_ready=on_ready, callback=self.metrics_collector)

    async def upload(
            self,
            location: str,
            video_mode: VideoMode | None = None,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None,
            *,
            is_live: bool | None = None,
    ) -> WorkerJob:
        self._not_sent(video_mode=video_mode, motion_detect=motion_detect, media_cache_seconds=media_cache_seconds)
        media = _file_media(location)
        return await self._submit(self._job(
            {'upload': {'contentType': media.content_type}}, [media], bool(is_live), params, roi, camera, fps,
            on_ready))

    async def upload_stream(
            self,
            stream: BinaryIO | AsyncIterable[bytes],
            mime_type: str,
            video_mode: VideoMode | None = None,
            is_live: bool | None = None,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        self._not_sent(video_mode=video_mode, motion_detect=motion_detect, media_cache_seconds=media_cache_seconds)
        media = _Media(lambda: stream, mime_type)
        return await self._submit(self._job(
            {'upload': {'contentType': mime_type}}, [media], bool(is_live), params, roi, camera, fps, on_ready))

    async def upload_stream_group(
            self,
            streams: list[BinaryIO],
            mime_types: list[str] | None = None,
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        if not streams:
            raise ValueError("upload requires at least one source")
        if mime_types is not None and len(mime_types) != len(streams):
            raise ValueError("mime_types must have the same length as streams")
        self._not_sent(media_cache_seconds=media_cache_seconds)
        types = mime_types or ['application/octet-stream'] * len(streams)
        media = [_Media(_opener(stream), mime_type) for stream, mime_type in zip(streams, types, strict=True)]
        return await self._submit(self._job(_upload_group(media), media, False, params, roi, camera, None, on_ready))

    async def upload_group(
            self,
            locations: list[str],
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        if not locations:
            raise ValueError("upload requires at least one source")
        self._not_sent(media_cache_seconds=media_cache_seconds)
        media = [_file_media(location) for location in locations]
        return await self._submit(self._job(_upload_group(media), media, False, params, roi, camera, None, on_ready))

    async def load_from(
            self,
            location: str,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            rtsp_force_non_compliant_url: bool | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        self._not_sent(motion_detect=motion_detect, media_cache_seconds=media_cache_seconds,
                       rtsp_force_non_compliant_url=rtsp_force_non_compliant_url)
        live = location.startswith(('rtsp://', 'rtmp://'))
        return await self._submit(self._job({'url': location}, [], live, params, roi, camera, fps, on_ready))

    async def load_from_group(
            self,
            locations: list[str],
            params: list[ComponentParams] | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        if not locations:
            raise ValueError("load_from requires at least one url")
        self._not_sent(media_cache_seconds=media_cache_seconds)
        group = {'group': [{'url': location} for location in locations]}
        return await self._submit(self._job(group, [], False, params, roi, camera, None, on_ready))

    async def load_asset(
            self,
            asset_uuid: str,
            params: list[ComponentParams] | None = None,
            motion_detect: MotionDetectConfig | None = None,
            roi: Area | None = None,
            camera: Camera | None = None,
            fps: str | None = None,
            media_cache_seconds: int | None = None,
            on_ready: Callable[[WorkerJob], None] | None = None
    ) -> WorkerJob:
        self._not_sent(motion_detect=motion_detect, media_cache_seconds=media_cache_seconds)
        return await self._submit(self._job({'assetUuid': asset_uuid}, [], False, params, roi, camera, fps, on_ready))

    # Session-era methods

    async def session(self) -> dict | None:
        return {'pipeline_api_url': self.pipeline_api_url}

    async def dev_mode_base_url(self) -> str:
        raise _no_session('dev_mode_base_url')

    async def dev_mode_pipeline_base_url(self) -> str:
        raise _no_session('dev_mode_pipeline_base_url')

    async def pipeline_get(self, url_path_and_query: str, accept: str | None = None,
                           timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_get')

    async def pipeline_post(self, url_path_and_query: str, accept: str | None = None, open_data: Callable | None = None,
                            content_type: str | None = None,
                            timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_post')

    async def pipeline_patch(self, url_path_and_query: str, accept: str | None = None, data: Any = None,
                             content_type: str | None = None,
                             timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_patch')

    async def pipeline_delete(self, url_path_and_query: str,
                              timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_delete')


def _no_session(method: str) -> NotImplementedError:
    return NotImplementedError(f'{method} needs a worker session; pipeline mode has none')


def _opener(stream: BinaryIO) -> Callable[[], BinaryIO]:
    return lambda: stream


def _file_media(location: str) -> _Media:
    return _Media(lambda: open(location, 'rb'), _guess_mime_type_from_location(location), os.path.getsize(location))


def _upload_group(media: list[_Media]) -> dict[str, Any]:
    return {'group': [{'upload': {'contentType': m.content_type}} for m in media]}


def _upload_keys(job: _SocketJob) -> list[int | None]:
    """The member index of each upload member, or None for a source that is one upload."""
    if 'upload' in job.source:
        return [None]
    group = job.source.get('group', [])
    return [i for i, member in enumerate(group) if 'upload' in member]


async def _combined_media(job: _SocketJob) -> list[bytes] | None:
    """The upload members' bytes when the job goes as one combined message, else None for chunked."""
    if not job.media or job.live or any(m.size is None for m in job.media):
        return None
    if sum(m.size or 0 for m in job.media) > settings.combined_upload_max_bytes:
        return None
    return [await asyncio.to_thread(_read_all, m) for m in job.media]


def _read_all(media: _Media) -> bytes:
    stream = media.open()
    assert not isinstance(stream, AsyncIterable)
    with stream:
        return stream.read()


def _combined_message(command: dict[str, Any], media: list[bytes]) -> bytes:
    if 'group' in command:
        sizes = iter(media)
        command = {**command, 'group': [
            {'upload': {**member['upload'], 'bytes': len(next(sizes))}} if 'upload' in member else member
            for member in command['group']
        ]}
    header = json.dumps(command).encode()
    return bytes([_COMBINED]) + struct.pack('>I', len(header)) + header + b''.join(media)


async def _pieces(stream: BinaryIO | AsyncIterable[bytes]) -> AsyncIterable[bytes]:
    if isinstance(stream, AsyncIterable):
        async for piece in stream:
            yield bytes(piece)
        return
    while piece := await asyncio.to_thread(stream.read, _CHUNK_BYTES):
        yield piece


def _checked_url(url: str) -> str:
    """Plain http only to a loopback host, so a token never crosses a network in clear."""
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https') or not parts.hostname:
        raise ValueError(f'the pipeline API url {url!r} must be an http or https url')
    if parts.scheme == 'http' and not _is_loopback(parts.hostname):
        raise ValueError(f'the pipeline API url {url!r} must use https unless its host is a loopback address')
    return url.rstrip('/')


def _is_loopback(host: str) -> bool:
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _retry_after(status: int, headers: Mapping[str, str] | None) -> float | None:
    if status != 503 or not headers:
        return None
    try:
        wait = float(headers['Retry-After'])
    except (KeyError, ValueError):
        return None
    # A floor, so that `Retry-After: 0` cannot turn an overload into a request storm.
    return max(wait, 1.0) if math.isfinite(wait) and wait >= 0 else None
