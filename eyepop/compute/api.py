import os

import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse
from eyepop.compute.status import wait_for_session

_compute_url = os.getenv("_COMPUTE_API_URL", "https://compute-api.staging.eyepop.xyz")

EYEPOP_USER_UUID = os.getenv("EYEPOP_USER_UUID", "")
EYEPOP_SECRET_KEY = os.getenv("EYEPOP_SECRET_KEY", "")
WAIT_FOR_SESSION_TIMEOUT = 10
WAIT_FOR_SESSION_INTERVAL = 1

def fetch_session_endpoint(user_uuid: str = EYEPOP_USER_UUID) -> str:
    session_response = fetch_new_compute_session(user_uuid)
    got_session = wait_for_session(session_response.session_endpoint, WAIT_FOR_SESSION_INTERVAL, WAIT_FOR_SESSION_TIMEOUT)
    if got_session:
        return session_response.session_endpoint
    else:
        raise Exception("Failed to fetch session endpoint")

def fetch_new_compute_session(user_uuid: str) -> ComputeApiSessionResponse:
    _compute_url = os.getenv("_COMPUTE_API_URL")
    compute_api_token = os.getenv("EYEPOP_SECRET_KEY")

    if _compute_url is None or compute_api_token is None:
        raise Exception("_COMPUTE_API_URL or EYEPOP_SECRET_KEY is not set")
    
    headers = {
        "Authorization": f"Bearer {EYEPOP_SECRET_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    
    response = requests.post(
        f"{_compute_url}/api/v1/session", 
        headers=headers,
        json={
            "user_uuid": user_uuid,
        }
    )
    response.raise_for_status()
    
    res = response.json()
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    
    return session_response.pipeline_url
