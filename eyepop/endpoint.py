import asyncio
import logging
import time
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, Awaitable

import aiohttp

from eyepop.client_session import ClientSession
from eyepop.metrics import MetricCollector
from eyepop.periodic import Periodic
from eyepop.request_tracer import RequestTracer
from eyepop.settings import settings

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')
log_metrics = logging.getLogger('eyepop.metrics')


if TYPE_CHECKING:
    from aiohttp import _RequestContextManager

async def response_check_with_error_body(response: aiohttp.ClientResponse):
    if not response.ok:
        message = await response.text()
        if message is None or len(message) == 0:
            message = response.reason
        raise aiohttp.ClientResponseError(
            request_info=response.request_info, # type: ignore
            history=response.history, # type: ignore
            status=response.status,
            message=message,
            headers=response.headers,
        )


class Endpoint(ClientSession):
    """Abstract EyePop Endpoint."""

    secret_key: str | None
    api_key: str | None
    provided_access_token: str | None
    eyepop_url: str
    token: dict[str, Any] | None
    expire_token_time: float | None
    compute_ctx: Any | None
    request_tracer: RequestTracer | None
    event_sender: Periodic | None
    retry_handlers: dict[int, Callable[[int, int], Awaitable[bool]]]
    client_session: aiohttp.ClientSession | None
    tasks: set[asyncio.Task]
    sem: asyncio.Semaphore
    metrics_collector: MetricCollector | None

    def __init__(
            self,
            secret_key: str | None,
            access_token: str | None,
            eyepop_url: str,
            job_queue_length: int,
            request_tracer_max_buffer: int,
            api_key: str | None = None
    ):
        self.secret_key = secret_key
        self.api_key = api_key
        if access_token is not None and access_token.lower().startswith("Bearer "):
            self.provided_access_token = access_token[len("Bearer "):]
        else:
            self.provided_access_token = access_token
        self.eyepop_url = eyepop_url
        self.token = None
        self.expire_token_time = None
        self.compute_ctx = None

        if api_key is not None:
            from eyepop.compute import ComputeContext
            self.compute_ctx = ComputeContext(
                compute_url=eyepop_url,
                api_key=api_key
            )
            log.debug("Compute API will be used, session will be fetched in _reconnect()")

        if request_tracer_max_buffer > 0:
            self.request_tracer = RequestTracer(max_events=request_tracer_max_buffer)
            self.event_sender = Periodic(self.send_trace_recordings, settings.send_trace_threshold_secs / 2)
        else:
            self.request_tracer = None
            self.event_sender = None

        self.retry_handlers = dict()
        if self.secret_key is not None:
            self.retry_handlers[401] = self._retry_401
        elif self.compute_ctx is not None:
            self.retry_handlers[401] = self._retry_401_compute
        self.retry_handlers[500] = self._retry_50x
        self.retry_handlers[502] = self._retry_50x
        self.retry_handlers[503] = self._retry_50x
        self.retry_handlers[504] = self._retry_50x

        self.client_session = None

        self.tasks = set()
        self.sem = asyncio.Semaphore(job_queue_length)

        if log_metrics.getEffectiveLevel() == logging.DEBUG:
            self.metrics_collector = MetricCollector()
        else:
            self.metrics_collector = None

    def add_retry_handler(self, status_code: int, handler: Callable[[int, int], Awaitable[bool]]):
        self.retry_handlers[status_code] = handler

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType], ) -> None:
        pass  # pragma: no cover

    async def __aenter__(self) -> "Endpoint":
        try:
            await self.connect()
        except aiohttp.ClientError as e:
            await self.disconnect()
            raise e

        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType], ) -> None:
        await self.disconnect()

    async def _authorization_header(self) -> str | None:
        access_token = await self.__get_access_token()
        if access_token is None:
            return None
        return f'Bearer {access_token}'

    async def disconnect(self, timeout: float | None = None) -> None:
        if timeout is None:
            await self._disconnect()
            await self._cleanup()
        else:
            try:
                async with asyncio.timeout(timeout):
                    await self._disconnect(timeout)
                    await self._cleanup()
            except asyncio.TimeoutError:
                log_requests.info(f"timeout after {timeout} seconds in disconnect, ignored")

    async def _cleanup(self) -> None:
        tasks = list(self.tasks)
        if len(tasks) > 0:
            await asyncio.gather(*tasks)

        if self.request_tracer and self.client_session:
            await self.event_sender.stop()
            if self.compute_ctx is None:
                await self.request_tracer.send_and_reset(f'{self.eyepop_url}/events', await self._authorization_header(),
                                                         None)

        if self.client_session:
            try:
                await self.client_session.close()
            except Exception as e:
                log.exception("error at disconnect", e)
            finally:
                self.client_session = None

        if self.metrics_collector:
            log_metrics.debug('endpoint disconnected, collected session metrics:')
            log_metrics.debug('total number of jobs: %d', self.metrics_collector.total_number_of_jobs)
            log_metrics.debug(f'max concurrent number of jobs: {self.metrics_collector.max_number_of_jobs_by_state}')
            log_metrics.debug(f'average wait time until state: {self.metrics_collector.get_average_times()}')

    async def connect(self):
        trace_configs = [self.request_tracer.get_trace_config()] if self.request_tracer else None
        self.client_session = aiohttp.ClientSession(
            raise_for_status=response_check_with_error_body,
            trace_configs=trace_configs
        )
        try:
            await self._reconnect()
        except Exception as e:
            await self.client_session.close()
            self.client_session = None
            raise e

        if self.event_sender is not None:
            await self.event_sender.start()

    async def session(self) -> dict | None:
        token = await self.__get_access_token()
        if token is None:
            return None
        return {
            'eyepopUrl': self.eyepop_url, 'accessToken': token,
            'validUntil': None if self.expire_token_time is None else self.expire_token_time * 1000
        }

    async def _reconnect(self):
        raise NotImplementedError

    async def _disconnect(self, timeout: float | None = None):
        raise NotImplementedError

    async def __get_access_token(self) -> str | None:
        if self.provided_access_token is not None:
            return self.provided_access_token
        if self.compute_ctx is not None:
            if self.compute_ctx.m2m_access_token:
                return self.compute_ctx.m2m_access_token
            else:
                log.debug("compute ctx m2m access token is None, fetching new token")
                authenticate_url = f'{self.compute_ctx.compute_url}/v1/auth/authenticate'
                api_auth_header = {
                    'Authorization': f'Bearer {self.compute_ctx.api_key}',
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                async with self.client_session.post(authenticate_url, headers=api_auth_header) as response:
                    response_json = await response.json()
                    self.compute_ctx.m2m_access_token = response_json['access_token']
                    log.debug(f"compute ctx m2m access token: {self.compute_ctx.m2m_access_token}")
                    return self.compute_ctx.m2m_access_token
        if self.secret_key is None:
            return None
        now = time.time()
        if self.token is None or self.expire_token_time < now:
            body = {'secret_key': self.secret_key}
            post_url = f'{self.eyepop_url}/authentication/token'
            log_requests.debug('before POST %s', post_url)
            async with self.client_session.post(post_url, json=body) as response:
                self.token = await response.json()
                self.expire_token_time = time.time() + self.token['expires_in'] - 60
            log_requests.debug('after POST %s expires_in=%d token_type=%s', post_url, self.token['expires_in'],
                               self.token['token_type'])
        log.debug('using access token, valid for at least %d seconds', self.expire_token_time - now)
        return self.token['access_token']

    def _task_done(self, task):
        self.tasks.discard(task)
        self.sem.release()

    async def _task_start(self, coro):
        await self.sem.acquire()
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._task_done)

    async def _retry_401(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('retry handler: after 401, about to retry with fresh access token')
            self.token = None
            self.expire_token_time = None
            return True

    async def _retry_401_compute(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 1:
            return False
        else:
            log_requests.debug('retry handler: after 401, about to refresh compute API token')
            if self.compute_ctx is None:
                log_requests.error('retry handler: compute_ctx is None, cannot refresh token')
                return False
            try:
                from eyepop.compute.api import refresh_compute_token
                self.compute_ctx = await refresh_compute_token(self.compute_ctx, self.client_session)
                log_requests.debug('retry handler: compute token refreshed successfully')
                return True
            except Exception as e:
                log_requests.error(f'retry handler: failed to refresh compute token: {e}')
                return False

    async def _retry_50x(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 3:
            return False
        else:
            wait_time = 2 ** (failed_attempts - 1)
            log_requests.info('retry handler: after %d, about to retry after %f seconds',
                              status_code, wait_time)
            await asyncio.sleep(wait_time)
            return True

    async def request_with_retry(
            self,
            method: str,
            url: str,
            accept: str | None = None,
            data: Any = None,
            content_type: str | None = None,
            timeout: aiohttp.ClientTimeout | None = None
    ) -> "_RequestContextManager":
        failed_attempts = 0
        while True:
            headers = {}
            authorization_header = await self._authorization_header()
            if authorization_header is not None:
                headers['Authorization'] = authorization_header
            if accept is not None:
                headers['Accept'] = accept
            if content_type is not None:
                headers['Content-Type'] = content_type
            try:
                log_requests.debug('before %s %s', method, url)
                if isinstance(data, Callable):
                    data = data()
                response = await self.client_session.request(method, url, headers=headers, data=data, timeout=timeout)
                log_requests.debug('after %s %s', method, url)
                return response
            except aiohttp.ClientResponseError as e:
                failed_attempts += 1
                if e.status not in self.retry_handlers:
                    raise e
                if not await self.retry_handlers[e.status](e.status, failed_attempts):
                    raise e
            except aiohttp.ClientConnectionError as e:
                failed_attempts += 1
                if 404 not in self.retry_handlers:
                    raise e
                if not await self.retry_handlers[404](404, failed_attempts):
                    raise e

    async def send_trace_recordings(self):
        if self.request_tracer is not None:
            await self.request_tracer.send_and_reset(f'{self.eyepop_url}/events',
                                                     await self._authorization_header(),
                                                     settings.send_trace_threshold_secs)
