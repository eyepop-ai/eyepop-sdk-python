import os

import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse

_compute_url = os.getenv("_COMPUTE_API_URL", "https://compute-api.staging.eyepop.xyz")


def fetch_worker_endpoint_url_from_compute(compute_x_token: str, account_uuid: str | None = None) -> str | None:
    headers = {
        "X-Token": compute_x_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    
    request_body = {}
    if account_uuid:
        request_body["account_uuid"] = account_uuid
    
    response = requests.post(
        f"{_compute_url}/api/v1/session", 
        headers=headers,
        json=request_body if request_body else None
    )
    response.raise_for_status()
    
    res = response.json()
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    return session_response.pipeline_url
