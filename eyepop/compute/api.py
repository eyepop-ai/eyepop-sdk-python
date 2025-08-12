
import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext
from eyepop.compute.status import wait_for_session


def fetch_session_endpoint(compute_config: ComputeContext) -> ComputeContext:
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
    
    response = requests.post(
        f"{compute_config.compute_url}/v1/session", 
        headers=headers,
        json={
            "user_uuid": compute_config.user_uuid,
        }
    )
    response.raise_for_status()
    
    res = response.json()
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    compute_config.session_endpoint = session_response.session_endpoint
    compute_config.session_uuid = session_response.session_uuid
    compute_config.pipeline_uuid = session_response.pipeline_uuid
    return compute_config