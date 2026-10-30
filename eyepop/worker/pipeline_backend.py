"""Runs a WorkerEndpoint's jobs on the pipeline API's sockets (SDK-24).

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
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import urljoin, urlsplit

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.api import refresh_compute_token
from eyepop.exceptions import (
    PipelineCommandException,
    PipelineSocketClosedException,
    PipelineSourceException,
    PopConfigurationException,
)
from eyepop.settings import settings
from eyepop.worker.worker_backend import WorkerBackend
from eyepop.worker.worker_jobs import WorkerJob, WorkerSource, _UploadSource
from eyepop.worker.worker_types import ComponentParams, Pop

if TYPE_CHECKING:
    from eyepop.worker.worker_endpoint import WorkerEndpoint

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
# The closes after which the backend opens a new socket at once; after any
# other close it opens one when the next job or set_pop needs it.
_RECONNECT_AT_ONCE = {(1008, 'credential_expired'), (1011, 'overloaded'), (1001, 'shutting_down')}
# The longest wait before such a reconnect, when one close follows another with no source played between.
_MAX_RECONNECT_DELAY_SECS = 30.0
# After the pipeline API refused another socket, the endpoint tries again only this much later.
_EXTRA_SOCKET_RETRY_SECS = 10.0
# What the socket's setSource does not take yet (AWSU-317).
_NOT_SENT = ('video_mode', 'motion_detect', 'media_cache_seconds')


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
    # The socket's ended count when the setSource went out.
    ended_before: int = 0

    def close(self, error: BaseException):
        for upload in self.uploads.values():
            upload.end()
        self.inbox.put_nowait(error)


class _Socket:
    """One upgraded socket: its open sources and commands by id, and the task that reads it."""

    def __init__(self, ws: aiohttp.ClientWebSocketResponse, wake: Callable[[], None]):
        self.ws = ws
        self._wake = wake
        self.sources: dict[str, _Source] = {}
        self.commands: dict[str, asyncio.Future] = {}
        self.closed = asyncio.Event()
        self.reader: asyncio.Task | None = None
        self.write_lock = asyncio.Lock()
        # The ids of sources sent on this socket that have neither ended nor been refused, as the
        # pipeline API counts them, cancelled ones included.
        self.held: set[str] = set()
        # The held sources that are live. A live source never ends, so nothing queued behind it plays.
        self.lives: set[str] = set()
        # The held sources the pipeline API took: only these end, so only these make room.
        self.accepted: set[str] = set()
        # The ends on this socket so far.
        self.ended = 0
        # How many sources the socket had taken when it last refused queue_full: while as many are
        # held, the next send waits. None once none is held.
        self.cap: int | None = None

    def freed(self, ended: bool = True):
        """A source ended or was refused, or the socket closed."""
        if ended:
            self.ended += 1
        if not self.held:
            self.cap = None
        self._wake()

    @property
    def has_room(self) -> bool:
        return self.cap is None or len(self.held) < self.cap

    @property
    def open(self) -> bool:
        return not self.closed.is_set() and not self.ws.closed

    async def send_json(self, message: dict[str, Any]):
        async with self.write_lock:
            await self.ws.send_str(json.dumps(message))

    async def send_bytes(self, data: bytes):
        async with self.write_lock:
            await self.ws.send_bytes(data)


class PipelineBackend(WorkerBackend):
    """Runs jobs on up to `pipeline_sockets` sockets to the pipeline API; connect() opens the first.

    A socket's pipeline plays its sources one after another, so a batch on one
    socket leaves the GPU mostly idle. While every open socket has a source and
    fewer than `pipeline_sockets` are open, the next job opens another; each job
    goes on the socket with the fewest open sources. Jobs go out in the order
    they were submitted, each as one source queued behind the ones before it
    on its socket. A source the pipeline API refuses queue_full,
    because the socket holds as many sources or as many image bytes as it
    takes, goes again once another source on the socket ended, so it may go
    after jobs submitted later. While the socket is full every send waits, and
    each end lets one go.
    """

    def __init__(self, endpoint: 'WorkerEndpoint'):
        self.endpoint = endpoint
        self.pipeline_api_url: str | None = None
        self._sockets: list[_Socket] = []
        self._send_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()
        self._reconnecting: asyncio.Task | None = None
        self._disconnecting = False
        self._token: str | None = None
        self._token_expires = 0.0
        self._next_id = 0
        self._warned: set[str] = set()
        self._no_extra_socket_until = 0.0
        self._room = asyncio.Event()
        self._open: dict[WorkerJob, tuple[_Socket, _Source]] = {}
        # Reconnects at once since a source last played.
        self._reconnects = 0

    def _wake(self):
        """Have the send waiting for room look again."""
        self._room.set()
        self._room = asyncio.Event()

    # Connection

    async def connect(self):
        self._disconnecting = False
        async with self._send_lock:
            await self._ensure_socket()

    async def disconnect(self, timeout: float | None = None):
        self._disconnecting = True
        tasks = list(self.endpoint.tasks)
        try:
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # Also when the endpoint's disconnect timeout cancels the wait for
            # a job that runs on, a live one: closing the socket ends it.
            if self._reconnecting is not None:
                self._reconnecting.cancel()
                await asyncio.gather(self._reconnecting, return_exceptions=True)
                self._reconnecting = None
            sockets, self._sockets = self._sockets, []
            for socket in sockets:
                await socket.ws.close()
                if socket.reader is not None:
                    await asyncio.gather(socket.reader, return_exceptions=True)

    async def session(self) -> dict | None:
        return {'pipeline_api_url': self.pipeline_api_url}

    def _open_sockets(self) -> list[_Socket]:
        return [socket for socket in self._sockets if socket.open]

    async def _ensure_socket(self) -> _Socket:
        """An open socket, or a new one with the last pop set on it. The caller holds the send lock."""
        async with self._connect_lock:
            sockets = self._open_sockets()
            return sockets[0] if sockets else await self._open_socket(patient=True)

    async def _socket_for_job(self, job: WorkerJob) -> _Socket:
        """The socket for the next job. The caller holds the send lock.

        A live job goes on a socket of its own, an idle one or a new one, which
        `pipeline_sockets` does not count. Any other job skips the sockets with a live
        source: a new one while every other socket has a source and fewer than
        `pipeline_sockets` are open, else the one with room and the fewest sources.
        """
        async with self._connect_lock:
            if _is_live(job.source):
                idle = next((socket for socket in self._open_sockets() if not socket.held), None)
                return idle or await self._open_socket(patient=True)
            sockets = [socket for socket in self._open_sockets() if not socket.lives]
            if not sockets:
                return await self._open_socket(patient=True)
            if (len(sockets) < settings.pipeline_sockets and all(socket.held for socket in sockets)
                    and time.monotonic() >= self._no_extra_socket_until):
                try:
                    return await self._open_socket(patient=False)
                except (aiohttp.ClientError, PipelineSocketClosedException, PopConfigurationException) as e:
                    # The open sockets take the job, and the jobs after it until the wait is over, so
                    # a full pipeline API is not asked for a socket once per job.
                    self._no_extra_socket_until = time.monotonic() + _EXTRA_SOCKET_RETRY_SECS
                    log_requests.info('could not open another pipeline API socket: %s', e)
            return min(sockets, key=lambda socket: (not socket.has_room, len(socket.held)))

    async def _open_socket(self, patient: bool) -> _Socket:
        """A new socket with the last pop set on it; `patient` waits through 503 with Retry-After."""
        if self.endpoint.client_session is None:
            raise PipelineSocketClosedException(None, 'the endpoint is not connected')
        url = await self._socket_url()
        ws = await self._upgrade(url, patient)
        socket = _Socket(ws, self._wake)
        socket.reader = asyncio.create_task(self._read(socket))
        self._sockets.append(socket)
        if self.endpoint.pop is not None:
            try:
                await self._set_pop_on(socket, self.endpoint.pop)
            except PopConfigurationException:
                # A socket whose pipeline does not run the pop runs no job.
                self._sockets.remove(socket)
                await socket.ws.close()
                raise
        return socket

    async def _socket_url(self) -> str:
        if self.pipeline_api_url is None:
            if self.endpoint.is_local_mode:
                base = self.endpoint.eyepop_url
            else:
                base = urljoin(self.endpoint.eyepop_url, await self._configured_pipeline_api_url())
            self.pipeline_api_url = _checked_url(base)
        return 'ws' + self.pipeline_api_url.removeprefix('http') + PIPELINE_PATH

    async def _configured_pipeline_api_url(self) -> str:
        client_session = self.endpoint.client_session
        assert client_session is not None
        config_url = f'{self.endpoint.eyepop_url.rstrip("/")}/v1/configs'
        compute_ctx = self.endpoint.compute_ctx
        if compute_ctx is not None and compute_ctx.account_uuid:
            config_url = f'{config_url}?account_uuid={compute_ctx.account_uuid}'
        headers = {}
        token = await self._pipeline_token()
        if token is not None:
            headers['Authorization'] = f'Bearer {token}'
        async with client_session.get(config_url, headers=headers) as response:
            config = await response.json()
        url = config.get('pipeline_api_url') if isinstance(config, dict) else None
        if not url:
            raise PopConfigurationException(
                'transient', f'{config_url} has no pipeline_api_url: this compute API serves no pipeline API')
        return url

    async def _pipeline_token(self, force: bool = False) -> str | None:
        """The user access token the socket sends; an API key is exchanged at the compute API and never sent."""
        endpoint = self.endpoint
        if endpoint.provided_access_token is not None:
            return endpoint.provided_access_token
        if endpoint.compute_ctx is None or not endpoint.compute_ctx.api_key:
            return None
        if force or self._token is None or self._token_expires - time.monotonic() < _REFRESH_BEFORE_EXPIRY_SECS:
            assert endpoint.client_session is not None
            endpoint.compute_ctx = await refresh_compute_token(endpoint.compute_ctx, endpoint.client_session)
            self._token = endpoint.compute_ctx.m2m_access_token
            self._token_expires = time.monotonic() + float(endpoint.compute_ctx.access_token_expires_in or 0)
        return self._token

    async def _upgrade(self, url: str, patient: bool = True) -> aiohttp.ClientWebSocketResponse:
        """Open the socket, waiting through 503 with Retry-After when `patient`, and exchanging the API key again after a 401."""
        client_session = self.endpoint.client_session
        assert client_session is not None
        deadline = time.monotonic() + settings.max_retry_time_secs
        rejected = False
        while True:
            token = await self._pipeline_token(force=rejected)
            headers = {'Authorization': f'Bearer {token}'} if token is not None else {}
            try:
                log_requests.debug('before upgrade %s', url)
                return await client_session.ws_connect(
                    url,
                    protocols=(SUBPROTOCOL,),
                    headers=headers,
                    heartbeat=settings.pipeline_heartbeat_secs,
                    max_msg_size=0,
                )
            except aiohttp.ClientResponseError as e:
                if (e.status == 401 and not rejected and self.endpoint.provided_access_token is None
                        and self._token is not None):
                    log_requests.debug('the pipeline API answered 401, exchanging the API key again')
                    rejected = True
                    continue
                wait = _retry_after(e.status, e.headers)
                if not patient or wait is None or time.monotonic() + wait > deadline:
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
        socket.freed()
        if socket in self._sockets:
            self._sockets.remove(socket)
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
            # Only the close that leaves none open reopens one; the others have sockets to run on.
            if not self._open_sockets():
                self._reconnecting = asyncio.create_task(self._reconnect_at_once())

    async def _reconnect_at_once(self):
        # The first reconnect is at once. Should the new socket be closed the
        # same way before a source plays on it, each next one waits twice as
        # long, so a pipeline API that keeps closing is not reconnected to in a loop.
        delay = 0.0 if self._reconnects == 0 else min(2.0 ** (self._reconnects - 1), _MAX_RECONNECT_DELAY_SECS)
        self._reconnects += 1
        try:
            if delay > 0:
                log_requests.info('reopening the pipeline API socket in %.0f s', delay)
                await asyncio.sleep(delay)
            async with self._send_lock:
                await self._ensure_socket()
        except Exception as e:
            # The next job or set_pop tries again and reports what fails.
            log_requests.info('could not reopen the pipeline API socket: %s', e)

    def _dispatch(self, socket: _Socket, message: dict[str, Any]):
        type_, id_ = message.get('type'), message.get('id')
        source = socket.sources.get(id_) if isinstance(id_, str) else None
        if type_ == 'end' or (type_ == 'error' and id_ in socket.held and message.get('error') != 'unknown_command'):
            # Also the end of a source its job already let go, a cancelled one: it frees room all the same.
            # A refused source never ends, so the refusal frees it. unknown_command answers a stopSource on a
            # pipeline API from before it, for a source still open.
            socket.held.discard(id_)
            socket.accepted.discard(id_)
            socket.lives.discard(id_)
            socket.freed(ended=type_ == 'end')
        elif type_ == 'result' and id_ in socket.held:
            socket.accepted.add(id_)
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
            if type_ == 'start':
                self._reconnects = 0
            if type_ == 'end':
                for upload in source.uploads.values():
                    upload.end()
            source.inbox.put_nowait(message)
        elif type_ == 'error':
            if id_ is None:
                log.warning('the pipeline API refused a message: %s %s', message.get('error'), message.get('message'))
            else:
                # A stopSource for a source that ended meanwhile, or one a pipeline API without
                # stopSource does not know: the job ended either way.
                log.debug('the pipeline API refused a command for %s: %s %s', id_, message.get('error'),
                          message.get('message'))

    # Pop

    async def set_pop(self, pop: Pop):
        """Set the pop on every open socket's pipeline; a pop the first refuses raises, and the last pop it took stays.

        shortcut: a refusal from a later socket, which the first took, leaves the earlier ones on the
        new pop; a refusal is the pop's own fault, so every socket answers it alike.
        """
        endpoint = self.endpoint
        if endpoint.client_session is None:
            endpoint.pop = pop
            return {'pop': pop.model_dump()}
        async with self._send_lock:
            sockets = self._open_sockets()
            if sockets:
                for socket in sockets:
                    await self._set_pop_on(socket, pop)
                endpoint.pop = pop
            else:
                previous, endpoint.pop = endpoint.pop, pop
                try:
                    await self._ensure_socket()
                except BaseException:
                    endpoint.pop = previous
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

    def accept(self, source: WorkerSource):
        """Warn once per endpoint for each option the socket does not take."""
        options = {
            'video_mode': source.video_mode,
            'motion_detect': source.motion_detect,
            'media_cache_seconds': source.media_cache_seconds,
            'rtsp_force_non_compliant_url': source.rtsp_force_non_compliant_url,
        }
        for name, value in options.items():
            if value is None or name in self._warned:
                continue
            self._warned.add(name)
            message = f'{name} is not sent to the pipeline API and has no effect in pipeline mode'
            if name in _NOT_SENT:
                message += ' (AWSU-317)'
            # The caller of the endpoint's method, past accept and that method.
            warnings.warn(message, UserWarning, stacklevel=4)

    async def cancelled(self, job: WorkerJob):
        """Stop the job's source on the pipeline, so that a cancelled live source does not run on."""
        opened = self._open.get(job)
        if opened is None:
            # It may be the send waiting for room.
            self._wake()
            return
        socket, source = opened
        source.inbox.put_nowait(_CANCELLED)
        if socket.open:
            try:
                await socket.send_json({'type': 'stopSource', 'id': source.id})
            except (ConnectionError, aiohttp.ClientError) as e:
                # The socket is closing, which stops the source too.
                log_requests.debug('could not stop a cancelled source: %s', e)

    async def run(self, job: WorkerJob):
        """Run the job's source to its end.

        A live source refused overloaded goes once more on a new socket. A
        source refused queue_full goes again once another source on the socket
        ended; with none of its other sources open there, or with an upload from
        the caller's own stream, which cannot be read twice, the refusal stands.
        """
        retried = False
        while True:
            sent = await self._set_source(job)
            if sent is None:
                return
            socket, source = sent
            uploader = None
            if source.uploads:
                uploader = asyncio.create_task(self._upload(socket, source, job))
            try:
                refused = await self._follow(job, source)
            finally:
                self._open.pop(job, None)
                socket.sources.pop(source.id, None)
                if uploader is not None:
                    uploader.cancel()
                    await asyncio.gather(uploader, return_exceptions=True)
            if refused is None:
                return
            if refused == 'queue_full':
                if not socket.accepted and socket.ended == source.ended_before:
                    raise PipelineCommandException('queue_full', 'the socket refused the source with none of its own open')
                if any(upload.location is None for upload in job.source.uploads):
                    raise PipelineCommandException('queue_full', 'the socket refused an upload from a stream, which cannot be sent again')
                # Sources still on their way count against the cap too, until they are taken or refused.
                socket.cap = min(len(socket.accepted), socket.cap or len(socket.accepted)) or None
                continue
            if retried or not _is_live(job.source):
                raise PipelineCommandException('overloaded', 'the pipeline API has no capacity for another live source')
            retried = True
            await socket.closed.wait()

    async def _set_source(self, job: WorkerJob) -> tuple[_Socket, _Source] | None:
        """Send the job's setSource, after every job submitted before it.

        The send lock is the first thing a job waits for, and asyncio.Lock wakes
        its waiters in order, so jobs go out in the order they were submitted.
        """
        async with self._send_lock:
            if job.cancelled:
                # Cancelled before its turn: never sent, so it takes no source on the pipeline.
                return None
            combined = await _combined_media(job.source)
            while True:
                socket = await self._socket_for_job(job)
                if job.cancelled:
                    return None
                if socket.open and socket.has_room:
                    break
                if socket.open:
                    # Full: each end on any socket lets one send look again, instead of every
                    # refused source sending its media at once.
                    await self._room.wait()
            source = _Source(self._new_id('source'))
            command = _set_source_command(source.id, job.source)
            if combined is None:
                for key in _upload_keys(job.source):
                    source.uploads[key] = _Upload()
            socket.sources[source.id] = source
            socket.held.add(source.id)
            if _is_live(job.source):
                socket.lives.add(source.id)
            source.ended_before = socket.ended
            self._open[job] = (socket, source)
            if combined is None:
                await socket.send_json(command)
            else:
                await socket.send_bytes(_combined_message(command, combined))
        return socket, source

    async def _follow(self, job: WorkerJob, source: _Source) -> str | None:
        """Push the source's predictions to the job until it ends; the refusal to retry on, overloaded or queue_full."""
        first = True
        while True:
            if job.cancelled:
                return None
            message = await source.inbox.get()
            if message is _CANCELLED:
                return None
            if isinstance(message, BaseException):
                raise message
            type_ = message.get('type')
            if type_ == 'prediction':
                if first:
                    job.callback.first_result(job)
                    first = False
                await job.push_message(message.get('prediction'))
            elif type_ == 'end':
                reason = message.get('reason')
                if reason == 'eos':
                    return None
                raise PipelineSourceException(source.id, reason or '', message.get('message', ''))
            elif type_ == 'error':
                if message.get('error') in ('overloaded', 'queue_full'):
                    return message['error']
                raise PipelineCommandException(message.get('error', ''), message.get('message', ''))
            elif type_ == 'dropped':
                log.debug('the pipeline API dropped %s predictions of a live source', message.get('count'))

    async def _upload(self, socket: _Socket, source: _Source, job: WorkerJob):
        """Send each upload member's bytes in chunks within its credit, then its uploaded."""
        try:
            for key, upload_source in zip(_upload_keys(job.source), job.source.uploads, strict=True):
                upload = source.uploads[key]
                stream = upload_source.open_stream()
                try:
                    async for piece in _pieces(stream):
                        while piece:
                            n = await upload.take(min(len(piece), _CHUNK_BYTES))
                            await socket.send_bytes(bytes([_CHUNK]) + piece[:n])
                            piece = piece[n:]
                finally:
                    # The job closes a file it opened; a caller's stream stays open.
                    if upload_source.location is not None and not isinstance(stream, AsyncIterable):
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


def _is_live(source: WorkerSource) -> bool:
    return bool(source.is_live) or any(url.startswith(('rtsp://', 'rtmp://')) for url in source.urls)


def _content_type(upload: _UploadSource) -> str:
    return upload.mime_type or 'application/octet-stream'


def _set_source_command(id: str, source: WorkerSource) -> dict[str, Any]:
    """The setSource that names the source and carries the options the socket takes."""
    command: dict[str, Any] = {'type': 'setSource', 'id': id}
    if source.uploads:
        if source.group:
            command['group'] = [{'upload': {'contentType': _content_type(u)}} for u in source.uploads]
        else:
            command['upload'] = {'contentType': _content_type(source.uploads[0])}
    elif source.asset_uuid is not None:
        command['assetUuid'] = source.asset_uuid
    elif source.group:
        command['group'] = [{'url': url} for url in source.urls]
    else:
        command['url'] = source.urls[0]
    if source.component_params is not None:
        command['params'] = TypeAdapter(list[ComponentParams]).dump_python(source.component_params)
    if source.roi is not None:
        command['roi'] = source.roi.model_dump(exclude_none=True)
    if source.camera is not None:
        command['camera'] = source.camera.model_dump(exclude_none=True)
    if source.fps is not None:
        command['fps'] = source.fps
    if source.is_live:
        command['live'] = True
    return command


def _upload_keys(source: WorkerSource) -> list[int | None]:
    """The member index of each upload, or None for a source that is one upload."""
    if not source.uploads:
        return []
    if not source.group:
        return [None]
    return list(range(len(source.uploads)))


async def _combined_media(source: WorkerSource) -> list[bytes] | None:
    """The uploads' bytes when the job goes as one combined message, else None for chunked.

    Only files the job opens itself go combined, when none is live and they add
    up to at most combined_upload_max_bytes.
    """
    uploads = source.uploads
    if not uploads or source.is_live or any(u.location is None for u in uploads):
        return None
    sizes = [os.path.getsize(u.location) for u in uploads if u.location is not None]
    if sum(sizes) > settings.combined_upload_max_bytes:
        return None
    return [await asyncio.to_thread(_read_all, u) for u in uploads]


def _read_all(upload: _UploadSource) -> bytes:
    stream = upload.open_stream()
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
