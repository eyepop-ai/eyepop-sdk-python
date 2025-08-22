import os

import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext
from eyepop.compute.status import wait_for_session


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
    
    get_response = requests.get(
        f"{compute_config.compute_url}/v1/sessions", 
        headers=headers,
    )
    get_response.raise_for_status()
    
    res = get_response.json()
    
    if not res or (isinstance(res, list) and len(res) == 0):
        try:
            post_response = requests.post(
                f"{compute_config.compute_url}/v1/sessions", 
                headers=headers,
            )
            post_response.raise_for_status()
            res = post_response.json()
        except Exception as e:
            raise Exception(f"No existing session and failed to create new one: {e}") from e
    
    is_arr = isinstance(res, list) and len(res) > 0
    if is_arr:
        res = res[0]
    
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    
    compute_config.session_endpoint = session_response.session_endpoint
    compute_config.session_uuid = session_response.session_uuid
    compute_config.access_token = session_response.access_token
    
    if not session_response.access_token or len(session_response.access_token.strip()) == 0:
        raise Exception("No access_token received from compute API session response. M2M authentication is not configured properly.")
    
    pipeline_id = session_response.pipelines[0]["pipeline_id"] if len(session_response.pipelines) > 0 else ""
    compute_config.pipeline_id = pipeline_id
    return compute_config