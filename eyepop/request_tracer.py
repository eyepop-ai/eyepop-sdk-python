import asyncio
import logging
import time
import uuid
import flatbuffers
from collections import deque
from types import SimpleNamespace
import importlib.metadata

import aiohttp
from aiohttp import TraceConfig, TraceRequestEndParams, TraceRequestExceptionParams, TraceRequestStartParams, \
    TraceResponseChunkReceivedParams, ClientConnectionError, ServerTimeoutError, ClientResponseError, \
    TraceRequestChunkSentParams

from eyepop.events.Event import EventStart, EventEnd, EventAddMethod, EventAddEventTimeEpochMs, EventAddXRequestId, \
    EventAddResult, EventAddStatus, EventAddHostIndex, EventAddWaitMs, EventAddProcessMs, EventAddPathIndex, \
    EventAddBodyBytesSent, EventAddBodyBytesReceived
from eyepop.events.Record import RecordStart, RecordEnd, RecordAddClientType, RecordAddHosts, \
    RecordAddClientVersion, RecordStartEventsVector, RecordAddEvents, RecordStartPathsVector, RecordAddPaths
from eyepop.events.ClientType import ClientType
from eyepop.events.Record import RecordStartHostsVector
from eyepop.events.Result import Result

log = logging.getLogger('eyepop.tracer')

__version__ = importlib.metadata.version('eyepop')

def delete_nth(d: deque, n: int):
    d.rotate(-n)
    d.popleft()
    d.rotate(n)


def is_event_started_ended_before(event: SimpleNamespace, threshold: float) -> (bool, bool):
    if event.start > threshold:
        return False, False
    last: float = None
    if hasattr(event, 'request_end'):
        last = event.request_end
    if hasattr(event, 'last_chunk_received') and (last is None or event.last_chunk_received > last):
        last = event.last_chunk_received
    if hasattr(event, 'exception') and (last is None or event.exception > last):
        last = event.exception
    return True, (last is not None) and (last <= threshold)


method_to_fb_enum = {
    'other': 0,
    'get': 1,
    'head': 2,
    'options': 3,
    'post': 4,
    'put': 5,
    'patch': 6,
    'delete': 7,
}

class RequestTracer():
    def __init__(self, max_events: int):
        self.events = deque(maxlen=max_events)

    def get_trace_config(self) -> TraceConfig:
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_chunk_sent.append(self.on_request_chunk_sent)
        trace_config.on_response_chunk_received.append(self.on_response_chunk_received)
        trace_config.on_request_exception.append(self.on_request_exception)
        trace_config.on_request_end.append(self.on_request_end)
        return trace_config

    async def send_and_reset(self, url: str, authorization_header: str, secs_to_mature: float | None):
        if secs_to_mature is None or secs_to_mature <= 0.0:
            matured_events = self.events
            self.events = deque(maxlen=self.events.maxlen)
        else:
            threshold = time.time() - secs_to_mature
            matured_events = deque()
            immature_events = deque()
            while len(self.events) > 0:
                event = self.events.pop()
                started_before, ended_before = is_event_started_ended_before(event, threshold)
                if ended_before:
                    matured_events.appendleft(event)
                elif started_before:
                    immature_events.appendleft(event)
                else:
                    immature_events.appendleft(event)
                    immature_events.extendleft(self.events)
                    break

            self.events = immature_events

        if len(matured_events) > 0:
            log.info('send_and_reset: %d (version: %s)', len(matured_events), __version__)
            builder = flatbuffers.Builder()
            host_to_index = {}
            hosts = []
            path_to_index = {}
            paths = []
            client_events = []
            for event in matured_events:
                if not event.host:
                    host_index = -1
                elif event.host in host_to_index.keys():
                    host_index = host_to_index[event.host]
                else:
                    host = builder.CreateString(event.host)
                    hosts.append(host)
                    host_index = len(hosts) - 1
                    host_to_index[event.host] = host_index

                if event.path in path_to_index.keys():
                    path_index = path_to_index[event.path]
                else:
                    path = builder.CreateString(event.path)
                    paths.append(path)
                    path_index = len(paths) - 1
                    path_to_index[event.path] = path_index

                if hasattr(event, 'x_request_id'):
                    x_request_id = builder.CreateString(event.x_request_id)
                else:
                    x_request_id = None
                EventStart(builder)
                EventAddMethod(builder, event.method)
                EventAddEventTimeEpochMs(builder, int(event.realtime * 1000))
                if x_request_id is not None:
                    EventAddXRequestId(builder, x_request_id)
                EventAddResult(builder, event.result)
                if hasattr(event, 'status'):
                    EventAddStatus(builder, event.status)

                EventAddHostIndex(builder, host_index)
                EventAddPathIndex(builder, path_index)

                if hasattr(event, 'exception'):
                    EventAddWaitMs(builder, round((event.exception - event.start) * 1000))
                if hasattr(event, 'request_end'):
                    EventAddWaitMs(builder, round((event.request_end - event.start) * 1000))
                    if hasattr(event, 'last_chunk_received'):
                        EventAddProcessMs(builder, round((event.last_chunk_received - event.request_end) * 1000))

                if hasattr(event, 'bytes_sent'):
                    EventAddBodyBytesSent(builder, event.bytes_sent)

                if hasattr(event, 'bytes_received'):
                    EventAddBodyBytesReceived(builder, event.bytes_received)

                client_event = EventEnd(builder)
                client_events.append(client_event)

            RecordStartHostsVector(builder, len(hosts))
            for host in reversed(hosts):
                builder.PrependSOffsetTRelative(host)
            hosts_vector = builder.EndVector()

            RecordStartPathsVector(builder, len(paths))
            for path in reversed(paths):
                builder.PrependSOffsetTRelative(path)
            paths_vector = builder.EndVector()

            RecordStartEventsVector(builder, len(client_events))
            for client_event in client_events:
                builder.PrependSOffsetTRelative(client_event)
            events_vector = builder.EndVector()

            client_version = builder.CreateString(__version__)

            RecordStart(builder)
            RecordAddClientType(builder, ClientType.python)
            RecordAddClientVersion(builder, client_version)
            RecordAddHosts(builder, hosts_vector)
            RecordAddPaths(builder, paths_vector)
            RecordAddEvents(builder, events_vector)
            record = RecordEnd(builder)
            builder.Finish(record)
            buf = builder.Output()

            async with aiohttp.ClientSession() as session:
                headers = {
                    'authorization': authorization_header,
                    'content-type': 'application/x-flatbuffers;schema=eyepop.events.Record'
                }

                async with session.post(url, data=buf, headers=headers) as resp:
                    pass


    async def on_request_start(self, session, trace_config_ctx: SimpleNamespace, params: TraceRequestStartParams):
        trace_config_ctx.x_request_id = uuid.uuid4().hex
        params.headers.add('X-Request-Id', trace_config_ctx.x_request_id)
        trace_config_ctx.realtime = time.time()
        trace_config_ctx.start = asyncio.get_event_loop().time()
        if params.url.port is not None:
            trace_config_ctx.host = f'{params.url.host}:{params.url.port}'
        else:
            trace_config_ctx.host = params.url.host
        trace_config_ctx.method = method_to_fb_enum[params.method.lower()]
        trace_config_ctx.path = params.url.path
        self.events.appendleft(trace_config_ctx)

    async def on_request_chunk_sent(self, session, trace_config_ctx: SimpleNamespace, params: TraceRequestChunkSentParams):
        bytes_sent = 0
        if hasattr(trace_config_ctx, 'bytes_sent'):
            bytes_sent = trace_config_ctx.bytes_sent
        trace_config_ctx.bytes_sent = bytes_sent + len(params.chunk)

    async def on_response_chunk_received(self, session, trace_config_ctx: SimpleNamespace, params: TraceResponseChunkReceivedParams):
        trace_config_ctx.last_chunk_received = asyncio.get_event_loop().time()
        bytes_received = 0
        if hasattr(trace_config_ctx, 'bytes_received'):
            bytes_received = trace_config_ctx.bytes_received
        trace_config_ctx.bytes_received = bytes_received + len(params.chunk)

    async def on_request_exception(self, session, trace_config_ctx: SimpleNamespace, params: TraceRequestExceptionParams):
        trace_config_ctx.exception = asyncio.get_event_loop().time()
        if isinstance(params.exception, ClientConnectionError):
            trace_config_ctx.result = Result.connection
        elif isinstance(params.exception, ServerTimeoutError):
            trace_config_ctx.result = Result.timeout
        elif isinstance(params.exception, ClientResponseError):
            trace_config_ctx.result = Result.status
        else:
            trace_config_ctx.result = Result.other


    async def on_request_end(self, session, trace_config_ctx: SimpleNamespace, params: TraceRequestEndParams):
        trace_config_ctx.request_end = asyncio.get_event_loop().time()
        trace_config_ctx.status = params.response.status
        trace_config_ctx.result = Result.success

