import asyncio
import logging

import aiohttp

from eyepop.compute.context import ComputeContext, PipelineStatus
from eyepop.compute.responses import ComputeApiSessionResponse
from eyepop.exceptions import ComputeHealthCheckException

log = logging.getLogger("eyepop.compute")


async def wait_for_session(
    compute_config: ComputeContext, client_session: aiohttp.ClientSession
) -> bool:
    timeout = compute_config.wait_for_session_timeout
    interval = compute_config.wait_for_session_interval

    # Session endpoint health check ALWAYS uses the JWT access_token
    if not compute_config.m2m_access_token or len(compute_config.m2m_access_token.strip()) == 0:
        raise ComputeHealthCheckException(
            "No access_token in compute_config. "
            "Cannot perform session health check. "
            "This should never happen - fetch_new_compute_session should have set it.",
            session_endpoint=compute_config.session_endpoint,
        )

    auth_header = f"Bearer {compute_config.m2m_access_token}"
    log.debug(
        f"Using JWT access_token for session health check at {compute_config.session_endpoint}/health"
    )

    headers = {
        "Authorization": auth_header,
        "Accept": "application/json",
    }

    health_url = f"{compute_config.session_endpoint}/health"
    log.debug(f"Waiting for session to be ready at: {health_url}")
    log.debug(f"Timeout: {timeout}s, Interval: {interval}s")

    end_time = asyncio.get_event_loop().time() + timeout
    last_message = "No message received"
    attempt = 0

    while asyncio.get_event_loop().time() < end_time:
        attempt += 1
        try:
            log.debug(f"Health check attempt {attempt}")

            async with client_session.get(health_url, headers=headers) as response:
                log.debug(f"Health check response status: {response.status}")

                if response.status == 200:
                    log.info("Session is ready (status 200)")
                    return True

                if response.status != 200:
                    last_message = f"Health check returned status {response.status}"
                    log.debug(last_message)
                    await asyncio.sleep(interval)
                    continue

                session_response = ComputeApiSessionResponse(**(await response.json()))
                status = session_response.session_status

                if status == PipelineStatus.RUNNING:
                    log.info("Session is running")
                    return True
                elif status == PipelineStatus.PENDING:
                    last_message = f"Session status: {status.value}"
                    log.debug(f"Session still pending/creating: {last_message}")
                    await asyncio.sleep(interval)
                    continue
                elif status in [
                    PipelineStatus.FAILED,
                    PipelineStatus.ERROR,
                    PipelineStatus.STOPPED,
                ]:
                    raise ComputeHealthCheckException(
                        f"Session in terminal state: {status.value}. Message: {session_response.session_message}",
                        session_endpoint=compute_config.session_endpoint,
                        last_status=status.value,
                    )
                else:
                    last_message = f"Session status: {status.value}"
                    log.debug(f"Unknown session status, continuing to wait: {last_message}")
                    await asyncio.sleep(interval)
                    continue

        except ComputeHealthCheckException:
            raise
        except aiohttp.ClientResponseError as e:
            last_message = f"HTTP {e.status}: {e.message}"
            log.debug(f"HTTP error during health check: {last_message}")
        except Exception as e:
            last_message = str(e)
            log.debug(f"Exception during health check: {last_message}")

        await asyncio.sleep(interval)

    log.error(f"Session timed out after {timeout}s. Last message: {last_message}")
    raise TimeoutError(f"Session timed out after {timeout}s. Last message: {last_message}")
