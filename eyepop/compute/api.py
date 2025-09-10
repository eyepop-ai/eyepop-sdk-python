import json
import logging
import os

import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext
from eyepop.compute.status import wait_for_session

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
log = logging.getLogger('eyepop.compute')

def fetch_session_endpoint(compute_config: ComputeContext | None = None) -> ComputeContext:
    if compute_config is None:
        compute_config = ComputeContext(
            compute_url=os.getenv("EYEPOP_URL", "https://compute.staging.eyepop.xyz"),
            secret_key=os.getenv("EYEPOP_SECRET_KEY", ""),
        )
    compute_context = fetch_new_compute_session(compute_config)
    
    got_session = wait_for_session(compute_context)
    if got_session:
        return compute_context
    else:
        raise Exception("Failed to fetch session endpoint")

def fetch_new_compute_session(compute_config: ComputeContext) -> ComputeContext:
    headers = {
        "Authorization": f"Bearer {compute_config.secret_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    log.debug(f"Fetching sessions from: {compute_config.compute_url}/v1/sessions")
    
    res = None
    need_new_session = False
    
    try:
        get_response = requests.get(
            f"{compute_config.compute_url}/v1/sessions", 
            headers=headers,
        )
        get_response.raise_for_status()
        res = get_response.json()
        log.info(json.dumps(res, indent=4)) 
        if not res:
            need_new_session = True
            log.debug("Response is empty/None, need to create new session")
        elif isinstance(res, list) and len(res) == 0:
            need_new_session = True
            log.debug("Response is empty list, need to create new session")
        elif isinstance(res, dict) and not res.get('session_uuid'):
            need_new_session = True
            log.debug("Response is dict without session_uuid, need to create new session")
            
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            need_new_session = True
            log.debug("GET /v1/sessions returned 404, will create new session")
        else:
            # Re-raise for other HTTP errors
            raise
    
    if need_new_session:
        try:
            log.debug(f"Creating new session via POST to: {compute_config.compute_url}/v1/sessions")
            post_response = requests.post(
                f"{compute_config.compute_url}/v1/sessions", 
                headers=headers,
            )
            post_response.raise_for_status()
            res = post_response.json()
            log.debug(f"POST /v1/sessions response: {res}")
        except Exception as e:
            raise Exception(f"No existing session and failed to create new one: {e}") from e
    
    is_arr = isinstance(res, list) and len(res) > 0
    if is_arr:
        log.debug(f"Response is array with {len(res)} items, using first one")  # pyright: ignore[reportArgumentType]
        res = res[0] if res else None
    
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    
    compute_config.session_endpoint = session_response.session_endpoint
    compute_config.session_uuid = session_response.session_uuid
    compute_config.access_token = session_response.access_token
    
    log.debug(f"Session endpoint: {session_response.session_endpoint}")
    log.debug(f"Session UUID: {session_response.session_uuid}")
    log.debug(f"Access token present: {bool(session_response.access_token)}")
    log.debug(f"Pipelines: {session_response.pipelines}")
    
    if not session_response.access_token or len(session_response.access_token.strip()) == 0:
        raise Exception("No access_token received from compute API session response. M2M authentication is not configured properly.")
    
    pipeline_id = session_response.pipelines[0]["pipeline_id"] if len(session_response.pipelines) > 0 else ""
    compute_config.pipeline_id = pipeline_id
    log.debug(f"Pipeline ID: {pipeline_id}")
    return compute_config