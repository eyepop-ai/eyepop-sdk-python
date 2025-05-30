import os

import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionRequest, ComputeApiSessionResponse
from eyepop.compute.status import wait_for_session

EYEPOP_USER_UUID = os.getenv("EYEPOP_USER_UUID", "")
EYEPOP_SECRET_KEY = os.getenv("EYEPOP_SECRET_KEY", "")
WAIT_FOR_SESSION_TIMEOUT = 10
WAIT_FOR_SESSION_INTERVAL = 1


def fetch_session_endpoint(account_uuid: str = "00000000-0000-0000-0000-000000000000") -> str:
    session_response = fetch_new_compute_session(account_uuid)
    got_session = wait_for_session(session_response.session_endpoint, WAIT_FOR_SESSION_INTERVAL, WAIT_FOR_SESSION_TIMEOUT)
    if got_session:
        return session_response.session_endpoint
    else:
        raise Exception("Failed to fetch session endpoint")

def fetch_new_compute_session(account_uuid: str) -> ComputeApiSessionResponse:
    _compute_url = os.getenv("_COMPUTE_API_URL")
    compute_api_token = os.getenv("EYEPOP_SECRET_KEY")

    if _compute_url is None or compute_api_token is None:
        raise Exception("_COMPUTE_API_URL or EYEPOP_SECRET_KEY is not set")
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {compute_api_token}"
    }
    
    request_body: ComputeApiSessionRequest = ComputeApiSessionRequest(account_uuid=account_uuid)
    
    response = requests.post(
        f"{_compute_url}/v1/session", 
        headers=headers,
        json=request_body.model_dump(),
        verify=False
    )

    try:
        response.raise_for_status()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)
    res_json = response.json()
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res_json)
    return session_response

