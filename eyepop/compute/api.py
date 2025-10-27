import logging

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext
from eyepop.compute.status import wait_for_session
from eyepop.exceptions import ComputeSessionException, ComputeTokenException

log = logging.getLogger('eyepop.compute')


async def fetch_session_endpoint(
    compute_config: ComputeContext,
    client_session: aiohttp.ClientSession
) -> ComputeContext:
    """
    Fetch or create a compute API session and wait for it to become ready.

    Args:
        compute_config: Configuration containing API key and compute URL
        client_session: Existing aiohttp session to reuse connections

    Returns:
        Updated ComputeContext with session details and access token

    Raises:
        ComputeSessionException: If session creation or health check fails
    """
    compute_context = await fetch_new_compute_session(compute_config, client_session)

    got_session = await wait_for_session(compute_context, client_session)
    if got_session:
        return compute_context

    raise ComputeSessionException(
        "Failed to fetch session endpoint",
        session_uuid=compute_context.session_uuid
    )


async def fetch_new_compute_session(
    compute_config: ComputeContext,
    client_session: aiohttp.ClientSession
) -> ComputeContext:
    """
    Fetch existing session or create new compute API session.

    Attempts to GET existing sessions first. If none exist or GET returns 404,
    creates a new session via POST.

    Args:
        compute_config: Configuration containing API key and compute URL
        client_session: Existing aiohttp session to reuse connections

    Returns:
        Updated ComputeContext with session endpoint and JWT access token

    Raises:
        ComputeSessionException: If session fetch/creation fails
    """
    headers = {
        "Authorization": f"Bearer {compute_config.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    sessions_url = f"{compute_config.compute_url}/v1/sessions"
    log.debug(f"Fetching sessions from: {sessions_url}")

    res = None
    need_new_session = False

    try:
        async with client_session.get(sessions_url, headers=headers) as get_response:
            if get_response.status == 404:
                need_new_session = True
                log.debug("GET /v1/sessions returned 404, will create new session")
            else:
                get_response.raise_for_status()
                res = await get_response.json()
                log.debug(f"GET /v1/sessions: {get_response.status}")

                if not res:
                    need_new_session = True
                    log.debug("Response is empty/None, need to create new session")
                elif isinstance(res, list) and len(res) == 0:
                    need_new_session = True
                    log.debug("Response is empty list, need to create new session")
                elif isinstance(res, dict) and not res.get('session_uuid'):
                    need_new_session = True
                    log.debug("Response is dict without session_uuid, need to create new session")

    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            need_new_session = True
            log.debug("GET /v1/sessions returned 404, will create new session")
        else:
            raise ComputeSessionException(
                f"Failed to fetch existing sessions: {e.message}",
            ) from e
    except Exception as e:
        raise ComputeSessionException(
            f"Unexpected error fetching sessions: {str(e)}"
        ) from e

    if need_new_session:
        try:
            log.debug(f"Creating new session via POST to: {sessions_url}")
            async with client_session.post(sessions_url, headers=headers) as post_response:
                post_response.raise_for_status()
                res = await post_response.json()
                log.debug(f"POST /v1/sessions response: {res}")
        except aiohttp.ClientResponseError as e:
            raise ComputeSessionException(
                f"Failed to create new session: {e.message}",
            ) from e
        except Exception as e:
            raise ComputeSessionException(
                f"No existing session and failed to create new one: {str(e)}"
            ) from e

    if isinstance(res, list):
        if len(res) > 0:
            log.debug(f"Response is array with {len(res)} items, using first one")
            res = res[0]
        else:
            res = None

    try:
        session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    except Exception as e:
        raise ComputeSessionException(
            f"Invalid session response format: {str(e)}"
        ) from e

    compute_config.session_endpoint = session_response.session_endpoint
    compute_config.session_uuid = session_response.session_uuid
    compute_config.m2m_access_token = session_response.access_token
    compute_config.access_token_expires_at = session_response.access_token_expires_at
    compute_config.access_token_expires_in = session_response.access_token_expires_in

    log.debug(f"Session endpoint: {session_response.session_endpoint}")
    log.debug(f"Session UUID: {session_response.session_uuid}")
    log.debug(f"Access token present: {bool(session_response.access_token)}")
    log.debug(f"Access token expires at: {session_response.access_token_expires_at}")
    log.debug(f"Access token expires in: {session_response.access_token_expires_in}s")
    log.debug(f"Pipelines: {session_response.pipelines}")

    if not session_response.access_token or len(session_response.access_token.strip()) == 0:
        raise ComputeSessionException(
            "No access_token received from compute API session response. "
            "M2M authentication is not configured properly.",
            session_uuid=compute_config.session_uuid
        )

    pipeline_id = session_response.pipelines[0]["pipeline_id"] if len(session_response.pipelines) > 0 else ""
    compute_config.pipeline_id = pipeline_id
    log.debug(f"Pipeline ID: {pipeline_id}")

    return compute_config


async def refresh_compute_token(
    compute_config: ComputeContext,
    client_session: aiohttp.ClientSession
) -> ComputeContext:
    """
    Refresh the JWT access token for a compute session.

    Args:
        compute_config: ComputeContext with session_uuid and api_key
        client_session: Existing aiohttp session to reuse connections

    Returns:
        Updated ComputeContext with new access token and expiry info

    Raises:
        ComputeTokenException: If token refresh fails
    """
    if not compute_config.session_uuid:
        raise ComputeTokenException(
            "Cannot refresh token: no session_uuid in compute_config"
        )

    if not compute_config.api_key:
        raise ComputeTokenException(
            "Cannot refresh token: no api_key in compute_config",
            session_uuid=compute_config.session_uuid
        )

    headers = {
        "Authorization": f"Bearer {compute_config.api_key}",
        "Accept": "application/json"
    }

    refresh_url = f"{compute_config.compute_url}/v1/auth/authenticate"
    log.info(f"Refreshing token at: {refresh_url}")

    try:
        async with client_session.post(refresh_url, headers=headers) as response:
            response.raise_for_status()
            token_response = await response.json()
            log.debug(f"Token refresh response: {token_response}")

            compute_config.m2m_access_token = token_response.get("access_token", "")
            compute_config.access_token_expires_at = token_response.get("expires_at", "")
            compute_config.access_token_expires_in = token_response.get("expires_in", 0)

            log.info(f"Token refreshed successfully, expires in: {compute_config.access_token_expires_in}s")
            return compute_config

    except aiohttp.ClientResponseError as e:
        log.error(f"Failed to refresh token: HTTP {e.status} - {e.message}")
        raise ComputeTokenException(
            f"Token refresh failed: HTTP {e.status} - {e.message}",
            session_uuid=compute_config.session_uuid
        ) from e
    except Exception as e:
        log.error(f"Failed to refresh token: {str(e)}")
        raise ComputeTokenException(
            f"Token refresh failed: {str(e)}",
            session_uuid=compute_config.session_uuid
        ) from e
