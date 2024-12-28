import asyncio
import logging
import time
from types import TracebackType
from typing import Optional, Type, Callable, Any

import aiohttp

from eyepop.client_session import ClientSession
from eyepop.metrics import MetricCollector
from eyepop.periodic import Periodic
from eyepop.request_tracer import RequestTracer

log = logging.getLogger('eyepop')
log_requests = logging.getLogger('eyepop.requests')
log_metrics = logging.getLogger('eyepop.metrics')

SEND_TRACE_THRESHOLD_SECS = 10.0


async def response_check_with_error_body(response: aiohttp.ClientResponse):
    if not response.ok:
        message = await response.text()
        if message is None or len(message) == 0:
            message = response.reason
        raise aiohttp.ClientResponseError(response.request_info, response.history, status=response.status,
                                          message=message, headers=response.headers, )


class Endpoint(ClientSession):
    """
    Abstract EyePop Endpoint.
    """

    def __init__(self, secret_key: str | None, access_token: str | None,
                 eyepop_url: str,
                 job_queue_length: int, request_tracer_max_buffer: int):
        if secret_key is None and access_token is None:
            raise ValueError("secret_key or access_token is required")
        self.secret_key = secret_key
        self.provided_access_token = access_token
        self.eyepop_url = eyepop_url
        self.token = None
        self.expire_token_time = None

        if request_tracer_max_buffer > 0:
            self.request_tracer = RequestTracer(max_events=request_tracer_max_buffer)
            self.event_sender = Periodic(self.send_trace_recordings, SEND_TRACE_THRESHOLD_SECS / 2)
        else:
            self.request_tracer = None
            self.event_sender = None

        self.retry_handlers = dict()
        if self.secret_key is not None:
            self.retry_handlers[401] = self._retry_401
        self.retry_handlers[500] = self._retry_50x
        self.retry_handlers[502] = self._retry_50x
        self.retry_handlers[503] = self._retry_50x
        self.retry_handlers[504] = self._retry_50x

        self.client_session = None
        self.task_group = None

        self.tasks = set()
        self.sem = asyncio.Semaphore(job_queue_length)

        if log_metrics.getEffectiveLevel() == logging.DEBUG:
            self.metrics_collector = MetricCollector()
        else:
            self.metrics_collector = None

    def add_retry_handler(self, status_code: int, handler: Callable[[int], bool]):
        self.retry_handlers[status_code] = handler

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException],
                 exc_tb: Optional[TracebackType], ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "WorkerEndpoint":
        try:
            await self.connect()
        except aiohttp.ClientError as e:
            await self.disconnect()
            raise e

        return self

    async def __aexit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException],
                        exc_tb: Optional[TracebackType], ) -> None:
        await self.disconnect()

    async def _authorization_header(self):
        return f'Bearer {await self.__get_access_token()}'

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
        self.client_session = aiohttp.ClientSession(raise_for_status=response_check_with_error_body,
                                                    trace_configs=trace_configs,
                                                    connector=aiohttp.TCPConnector(limit_per_host=8))
        try:
            await self._reconnect()
        except Exception as e:
            await self.client_session.close()
            self.client_session = None
            raise e

        if self.event_sender is not None:
            await self.event_sender.start()

    async def session(self) -> dict:
        token = await self.__get_access_token()
        session = {
            'eyepopUrl': self.eyepop_url, 'accessToken': token,
            'validUntil': None if self.expire_token_time is None else self.expire_token_time * 1000
        }

        return session

    async def _reconnect(self):
        raise NotImplemented

    async def _disconnect(self, timeout: float | None = None):
        raise NotImplemented

    async def __get_access_token(self):
        if self.provided_access_token is not None:
            return self.provided_access_token
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

    async def _retry_50x(self, status_code: int, failed_attempts: int) -> bool:
        if failed_attempts > 3:
            return False
        else:
            wait_time = 2 ** (failed_attempts - 1)
            log_requests.info('retry handler: after %d, about to retry after %f seconds',
                              status_code, wait_time)
            await asyncio.sleep(wait_time)
            return True

    async def request_with_retry(self, method: str, url: str, accept: str | None = None,
                                 data: Any = None, content_type: str | None = None,
                                 timeout: aiohttp.ClientTimeout | None = None) -> "_RequestContextManager":
        failed_attempts = 0
        while True:
            headers = {'Authorization': await self._authorization_header()}
            if accept is not None:
                headers['Accept'] = accept
            if content_type is not None:
                headers['Content-Type'] = content_type
            try:
                log_requests.debug('before %s %s', method, url)
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
            await self.request_tracer.send_and_reset(f'{self.eyepop_url}/events', await self._authorization_header(),
                                                     SEND_TRACE_THRESHOLD_SECS)
