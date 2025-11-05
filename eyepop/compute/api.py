import json
import logging

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.context import ComputeContext
from eyepop.compute.responses import ComputeApiSessionResponse
from eyepop.compute.status import wait_for_session
from eyepop.exceptions import ComputeSessionException, ComputeTokenException

log = logging.getLogger("eyepop.compute")


async def fetch_session_endpoint(
    compute_ctx: ComputeContext, client_session: aiohttp.ClientSession
) -> ComputeContext:
    """Fetch or create a compute API session and wait for it to become ready.

    Args:
        compute_ctx: ComputeContext containing API key and compute URL
        client_session: Existing aiohttp session to reuse connections

    Returns:
        Updated ComputeContext with session details and access token

    Raises:
        ComputeSessionException: If session creation or health check fails
    """
    compute_context = await fetch_new_compute_session(compute_ctx, client_session)

    got_session = await wait_for_session(compute_context, client_session)
    if got_session:
        return compute_context

    raise ComputeSessionException(
        "Failed to fetch session endpoint", session_uuid=compute_context.session_uuid
    )


async def fetch_new_compute_session(
    compute_ctx: ComputeContext, client_session: aiohttp.ClientSession
) -> ComputeContext:
    headers = {
        "Authorization": f"Bearer {compute_ctx.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    sessions_url = f"{compute_ctx.compute_url}/v1/sessions"
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
                elif isinstance(res, dict) and not res.get("session_uuid"):
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
        raise ComputeSessionException(f"Unexpected error fetching sessions: {str(e)}") from e

    if need_new_session:
        try:
            log.debug(f"Creating new session via POST to: {sessions_url}")
            async with client_session.post(sessions_url, headers=headers) as post_response:
                post_response.raise_for_status()
                res = await post_response.json()
                log.debug(f"POST /v1/sessions response: {post_response.status}")
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
            log.warning(f"Session response gave multiple {len(res)} items, using first one")
            res = res[0]
        else:
            res = None

    try:
        session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    except Exception as e:
        raise ComputeSessionException(f"Invalid session response format: {str(e)}") from e

    compute_ctx.session_endpoint = session_response.session_endpoint
    compute_ctx.session_uuid = session_response.session_uuid
    compute_ctx.m2m_access_token = session_response.access_token
    compute_ctx.access_token_expires_at = session_response.access_token_expires_at
    compute_ctx.access_token_expires_in = session_response.access_token_expires_in
    pipeline_id = (
        session_response.pipelines[0]["pipeline_id"] if len(session_response.pipelines) > 0 else ""
    )
    compute_ctx.pipeline_id = pipeline_id

    debug_obj = {
        "session_endpoint": session_response.session_endpoint,
        "session_uuid": session_response.session_uuid,
        "m2m_access_token": session_response.access_token,
        "m2m_access_token_expires_at": session_response.access_token_expires_at,
        "m2m_access_token_expires_in": session_response.access_token_expires_in,
        "pipeline_id": pipeline_id,
        "pipelines": session_response.pipelines,
    }
    log.debug(json.dumps(debug_obj, indent=4))

    if not session_response.access_token or len(session_response.access_token.strip()) == 0:
        raise ComputeSessionException(
            "No M2M access_token received from compute API session response. "
            "M2M authentication is not configured properly.",
            session_uuid=compute_ctx.session_uuid,
        )

    return compute_ctx


async def refresh_compute_token(
    compute_ctx: ComputeContext, client_session: aiohttp.ClientSession
) -> ComputeContext:
    if not compute_ctx.api_key:
        raise ComputeTokenException(
            "Cannot refresh token: no api_key in compute_ctx",
            session_uuid=compute_ctx.session_uuid,
        )

    headers = {"Authorization": f"Bearer {compute_ctx.api_key}", "Accept": "application/json"}

    refresh_url = f"{compute_ctx.compute_url}/v1/auth/authenticate"
    log.info(f"Refreshing token at: {refresh_url}")

    try:
        async with client_session.post(refresh_url, headers=headers) as response:
            response.raise_for_status()
            token_response = await response.json()
            log.debug(f"Token refresh response: {token_response}")

            compute_ctx.m2m_access_token = token_response.get("access_token", "")
            compute_ctx.access_token_expires_at = token_response.get("expires_at", "")
            compute_ctx.access_token_expires_in = token_response.get("expires_in", 0)

            log.debug(
                f"Token refreshed successfully, expires in: {compute_ctx.access_token_expires_in}s"
            )
            return compute_ctx

    except aiohttp.ClientResponseError as e:
        log.error(f"Failed to refresh token: HTTP {e.status} - {e.message}")
        raise ComputeTokenException(
            f"Token refresh failed: HTTP {e.status} - {e.message}",
            session_uuid=compute_ctx.session_uuid,
        ) from e
    except Exception as e:
        log.error("Failed to refresh token")
        log.debug(str(e))
        raise ComputeTokenException(
            f"Token refresh failed: {str(e)}", session_uuid=compute_ctx.session_uuid
        ) from e
