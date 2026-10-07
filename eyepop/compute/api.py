import json
import logging
import re
from typing import Any

import aiohttp
from pydantic import TypeAdapter

from eyepop.compute.context import ComputeContext, first_pipeline_id
from eyepop.compute.responses import ComputeApiSessionResponse
from eyepop.compute.status import wait_for_session
from eyepop.exceptions import ComputeSessionException, ComputeTokenException

log = logging.getLogger("eyepop.compute")

_MAX_SESSION_NAME_LENGTH = 63


async def fetch_session_endpoint(
    compute_ctx: ComputeContext,
    client_session: aiohttp.ClientSession,
    permanent_session_uuid: str | None,
    is_local_mode: bool = False,
) -> ComputeContext:
    """Fetch or create a compute API session, then poll until ready."""
    if permanent_session_uuid is None:
        compute_context = await fetch_new_compute_session(compute_ctx, client_session)

        if not is_local_mode:
            got_session = await wait_for_session(compute_context, client_session)
        else:
            got_session = True

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
    need_new_session = compute_ctx.pop is not None

    if not need_new_session:
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
                    elif isinstance(res, list):
                        res = [
                            s for s in res
                            if isinstance(s, dict) and _can_attach(s, compute_ctx.session_name, compute_ctx.account_uuid)
                        ]
                        if not res:
                            need_new_session = True
                    elif isinstance(res, dict):
                        if not res.get("session_uuid") or not _can_attach(
                            res, compute_ctx.session_name, compute_ctx.account_uuid
                        ):
                            need_new_session = True

        except aiohttp.ClientResponseError as e:
            raise ComputeSessionException(
                f"Failed to fetch existing sessions: {e.message}",
            ) from e
        except Exception as e:
            raise ComputeSessionException(f"Unexpected error fetching sessions: {str(e)}") from e

    if need_new_session:
        try:
            body = {}
            if compute_ctx.account_uuid:
                body["account_uuid"] = compute_ctx.account_uuid
            if compute_ctx.session_name:
                body["session_name"] = compute_ctx.session_name
            if compute_ctx.pipeline_image:
                body["pipeline_image"] = compute_ctx.pipeline_image
            if compute_ctx.pipeline_version:
                body["pipeline_version"] = compute_ctx.pipeline_version
            if compute_ctx.pop is not None:
                body["pop"] = compute_ctx.pop

            query = "wait=true"
            if compute_ctx.pop is not None:
                query = f"{query}&transient=true"

            async with client_session.post(
                f'{sessions_url}?{query}',
                headers=headers,
                json=body if body else None,
            ) as post_response:
                log.debug(f"POST /v1/sessions - status: {post_response.status}")
                if post_response.status >= 400:
                    reason = await _error_message(post_response)
                    raise ComputeSessionException(
                        f"Failed to create new session: HTTP {post_response.status} - {reason}",
                    )
                res = await post_response.json()
        except ComputeSessionException:
            raise
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


async def _error_message(response: aiohttp.ClientResponse) -> str:
    """The reason an error response gives, from its `error.message` when it has one.

    The HTTP reason phrase alone hides why the compute API refused, for instance
    that it could not derive account_uuid from the credential.
    """
    try:
        text = await response.text()
    except Exception:
        return response.reason or ""
    try:
        payload = json.loads(text)
    except ValueError:
        return text or response.reason or ""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str) and error:
            return error
        if payload.get("message"):
            return str(payload["message"])
    return text or response.reason or ""


def _can_attach(session: dict, requested_name: str, account_uuid: str | None = None) -> bool:
    """Whether a listed session may be reused for this caller.

    Never a persistent session, and with a requested account never a session
    of another account, or one that does not say which it runs for. With no
    requested name any transient will do; with one, only a transient that
    answers to that name, as the sessions API itself matches it: by display
    name or session name, as given or sanitized.
    """
    if session.get("persistent"):
        return False
    if account_uuid and session.get("account_uuid") != account_uuid:
        return False
    if not requested_name:
        return True
    names = {session.get("display_name"), session.get("session_name")}
    sanitized = _sanitize_session_name(requested_name)
    return requested_name in names or bool(sanitized and sanitized in names)


def _sanitize_session_name(name: str) -> str:
    """The display name the sessions API stores for a requested session name."""
    s = name.strip().lower().replace(" ", "-").replace("_", "-")
    s = re.sub(r"[^a-z0-9-]", "", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s[:_MAX_SESSION_NAME_LENGTH].rstrip("-")


def _compute_context_from_response(compute_ctx: ComputeContext, res: dict | None | Any):
    try:
        session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    except Exception as e:
        raise ComputeSessionException(f"Invalid session response format: {str(e)}") from e

    compute_ctx.session_endpoint = session_response.session_endpoint
    compute_ctx.session_uuid = session_response.session_uuid
    # A requested name is what later requests must keep asking for. Only an
    # unnamed caller takes the server's name, which pins it to this session.
    if session_response.session_name and not compute_ctx.session_name:
        compute_ctx.session_name = session_response.session_name
    compute_ctx.m2m_access_token = session_response.access_token
    compute_ctx.access_token_expires_at = session_response.access_token_expires_at
    compute_ctx.access_token_expires_in = session_response.access_token_expires_in
    pipeline_id = first_pipeline_id(session_response.pipelines)
    compute_ctx.pipeline_id = pipeline_id
    compute_ctx.pipeline_owned = bool(compute_ctx.pop is not None and pipeline_id)

    debug_obj = {
        "session_endpoint": session_response.session_endpoint,
        "session_uuid": session_response.session_uuid,
        "m2m_access_token": session_response.access_token,
        "m2m_access_token_expires_at": session_response.access_token_expires_at,
        "m2m_access_token_expires_in": session_response.access_token_expires_in,
        "pipeline_id": pipeline_id,
        "pipeline_owned": compute_ctx.pipeline_owned,
        "pipelines": session_response.pipelines,
    }
    log.debug(json.dumps(debug_obj, indent=4))

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
