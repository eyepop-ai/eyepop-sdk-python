import logging
from typing import Any

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.context import ComputeContext
from eyepop.compute.responses import ComputeApiSessionResponse
from eyepop.compute.status import wait_for_session
from eyepop.exceptions import ComputeSessionException, ComputeTokenException

log = logging.getLogger("eyepop.compute")


async def fetch_session_endpoint(
    compute_ctx: ComputeContext,
    client_session: aiohttp.ClientSession,
    permanent_session_uuid: str | None
) -> ComputeContext:
    """Fetch or create a compute API session, then poll until ready."""
    if permanent_session_uuid is None:
        compute_context = await fetch_new_compute_session(compute_ctx, client_session)

        got_session = await wait_for_session(compute_context, client_session)
        if got_session:
            return compute_context
    else:
        return await fetch_permanent_compute_session(
            compute_ctx=compute_ctx,
            client_session=client_session,
            permanent_session_uuid=permanent_session_uuid,
        )
    raise ComputeSessionException(
        "Failed to fetch session endpoint", session_uuid=compute_context.session_uuid
    )


async def fetch_new_compute_session(
    compute_ctx: ComputeContext,
    client_session: aiohttp.ClientSession
) -> ComputeContext:
    headers = {
        "Authorization": f"Bearer {compute_ctx.api_key}",
        "Accept": "application/json",
    }

    sessions_url = f"{compute_ctx.compute_url}/v1/sessions"

    res = None
    need_new_session = False

    try:
        async with client_session.get(sessions_url, headers=headers) as get_response:
            log.debug(f"GET /v1/sessions - status: {get_response.status}")
            if get_response.status == 404:
                need_new_session = True
            else:
                get_response.raise_for_status()
                res = await get_response.json()

                if not res:
                    need_new_session = True
                elif isinstance(res, list) and len(res) == 0:
                    need_new_session = True
                elif isinstance(res, dict) and not res.get("session_uuid"):
                    need_new_session = True

    except aiohttp.ClientResponseError as e:
        if e.status == 404:
            need_new_session = True
        else:
            raise ComputeSessionException(
                f"Failed to fetch existing sessions: {e.message}",
            ) from e
    except Exception as e:
        raise ComputeSessionException(f"Unexpected error fetching sessions: {str(e)}") from e

    if need_new_session:
        try:
            body = {}
            if compute_ctx.pipeline_image:
                body["pipeline_image"] = compute_ctx.pipeline_image
            if compute_ctx.pipeline_version:
                body["pipeline_version"] = compute_ctx.pipeline_version
            if compute_ctx.session_opts:
                body.update(compute_ctx.session_opts)

            async with client_session.post(
                f'{sessions_url}?wait=true',
                headers=headers,
                json=body if body else None,
            ) as post_response:
                post_response.raise_for_status()
                res = await post_response.json()
                log.debug(f"POST /v1/sessions - status: {post_response.status}")
        except aiohttp.ClientResponseError as e:
            raise ComputeSessionException(
                f"Failed to create new session: {e.message}",
            ) from e
        except Exception as e:
            raise ComputeSessionException(
                f"No existing session and failed to create new one: {str(e)}"
            ) from e

    if isinstance(res, list):
        if len(res) > 1:
            log.warning(f"Session response gave multiple {len(res)} items, using first one")
        if len(res) > 0:
            res = res[0]
        else:
            res = None

    _compute_context_from_response(compute_ctx, res)

    return compute_ctx


def _compute_context_from_response(compute_ctx: ComputeContext, res: dict | None | Any):
    try:
        session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    except Exception as e:
        raise ComputeSessionException(f"Invalid session response format: {str(e)}") from e

    compute_ctx.session_endpoint = session_response.session_endpoint
    compute_ctx.session_uuid = session_response.session_uuid
    compute_ctx.m2m_access_token = session_response.access_token
    compute_ctx.access_token_expires_at = session_response.access_token_expires_at
    compute_ctx.access_token_expires_in = session_response.access_token_expires_in
    pipeline_id = ""

    if len(session_response.pipelines) > 0:
        pipeline_id = session_response.pipelines[0].get("id", None)
        if not pipeline_id:
            pipeline_id = session_response.pipelines[0].get("pipeline_id", "")

    compute_ctx.pipeline_id = pipeline_id

    if not session_response.access_token or len(session_response.access_token.strip()) == 0:
        raise ComputeSessionException(
            "No M2M access_token received from compute API session response. "
            "M2M authentication is not configured properly.",
            session_uuid=compute_ctx.session_uuid,
        )


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

    try:
        async with client_session.post(refresh_url, headers=headers) as response:
            response.raise_for_status()
            token_response = await response.json()
            log.debug(f"POST /v1/auth/authenticate - status: {response.status}")

            compute_ctx.m2m_access_token = token_response.get("access_token", "")
            compute_ctx.access_token_expires_at = token_response.get("expires_at", "")
            compute_ctx.access_token_expires_in = token_response.get("expires_in", 0)

            return compute_ctx

    except aiohttp.ClientResponseError as e:
        raise ComputeTokenException(
            f"Token refresh failed: HTTP {e.status} - {e.message}",
            session_uuid=compute_ctx.session_uuid,
        ) from e
    except Exception as e:
        raise ComputeTokenException(
            f"Token refresh failed: {str(e)}", session_uuid=compute_ctx.session_uuid
        ) from e


async def fetch_permanent_compute_session(
    compute_ctx: ComputeContext,
    client_session: aiohttp.ClientSession,
    permanent_session_uuid: str
) -> ComputeContext:
    headers = {
        "Authorization": f"Bearer {compute_ctx.api_key}",
        "Accept": "application/json",
    }

    session_url = f"{compute_ctx.compute_url}/v1/sessions/{permanent_session_uuid}"

    try:
        async with client_session.get(session_url, headers=headers) as get_response:
            get_response.raise_for_status()
            res = await get_response.json()
            log.debug(f"GET /v1/sessions/{permanent_session_uuid} - status: {get_response.status}")
            _compute_context_from_response(compute_ctx, res)
            return compute_ctx
    except aiohttp.ClientResponseError as e:
        raise ComputeSessionException(
            f"Failed to fetch existing sessions: {e.message}",
        ) from e
    except Exception as e:
        raise ComputeSessionException(f"Unexpected error fetching sessions: {str(e)}") from e
