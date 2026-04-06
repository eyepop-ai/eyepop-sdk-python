import asyncio
import logging

import aiohttp

from eyepop.compute.context import ComputeContext
from eyepop.exceptions import ComputeHealthCheckException

log = logging.getLogger("eyepop.compute")


async def wait_for_session(
    compute_config: ComputeContext, client_session: aiohttp.ClientSession
) -> bool:
    timeout = compute_config.wait_for_session_timeout
    interval = compute_config.wait_for_session_interval

    if not compute_config.m2m_access_token or len(compute_config.m2m_access_token.strip()) == 0:
        raise ComputeHealthCheckException(
            "No access_token in compute_config. "
            "Cannot perform session health check. "
            "This should never happen - fetch_new_compute_session should have set it.",
            session_endpoint=compute_config.session_endpoint,
        )

    headers = {
        "Authorization": f"Bearer {compute_config.m2m_access_token}",
        "Accept": "application/json",
    }

    health_url = f"{compute_config.session_endpoint}/health"
    log.debug(f"Waiting for session ready: {health_url} (timeout={timeout}s, interval={interval}s)")

    end_time = asyncio.get_event_loop().time() + timeout
    last_message = "No message received"
    attempt = 0

    while asyncio.get_event_loop().time() < end_time:
        attempt += 1
        try:
            async with client_session.get(health_url, headers=headers) as response:
                log.debug(f"GET /health - status: {response.status} (attempt {attempt})")

                if response.status == 200:
                    return True

                last_message = f"Health check returned status {response.status}"
                await asyncio.sleep(interval)
                continue

        except ComputeHealthCheckException:
            raise
        except aiohttp.ClientResponseError as e:
            last_message = f"HTTP {e.status}: {e.message}"
            log.debug(f"GET /health - error: {last_message} (attempt {attempt})")
        except Exception as e:
            last_message = str(e)
            log.debug(f"GET /health - error: {last_message} (attempt {attempt})")

        await asyncio.sleep(interval)

    log.error(f"Session timed out after {timeout}s. Last message: {last_message}")
    raise TimeoutError(f"Session timed out after {timeout}s. Last message: {last_message}")
