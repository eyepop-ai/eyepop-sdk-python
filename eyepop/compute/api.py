import logging
import os
import requests
from pydantic import TypeAdapter

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext
from eyepop.compute.status import wait_for_session

def fetch_session_endpoint(compute_config: ComputeContext = None) -> ComputeContext:
    if compute_config is None:
        compute_config = ComputeContext(
            compute_url=os.getenv("EYEPOP_URL", "https://compute.staging.eyepop.xyz"),
            secret_key=os.getenv("EYEPOP_SECRET_KEY", ""),
        )
    compute_context = fetch_new_compute_session(compute_config)

    # Skip health check for now - session endpoint may not have /health endpoint
    # got_session = wait_for_session(compute_context)
    # if got_session:
    #     return compute_context
    # else:
    #     raise Exception("Failed to fetch session endpoint")
    
    # Just return the context with session data
    return compute_context

def fetch_new_compute_session(compute_config: ComputeContext) -> ComputeContext:
    headers = {
        "Authorization": f"Bearer {compute_config.secret_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    logging.debug(f"Fetching session from: {compute_config.compute_url}/v1/sessions")
    logging.debug(f"Using secret key: {compute_config.secret_key[:10]}..." if compute_config.secret_key else "No secret key!")
    
    # First try GET to check for existing session
    get_response = requests.get(
        f"{compute_config.compute_url}/v1/sessions", 
        headers=headers,
    )
    get_response.raise_for_status()
    
    res = get_response.json()
    
    # If GET returns empty, try to create a new session with POST
    if not res or (isinstance(res, list) and len(res) == 0):
        logging.info("No existing sessions found, creating a new one...")
        try:
            post_response = requests.post(
                f"{compute_config.compute_url}/v1/sessions", 
                headers=headers,
            )
            post_response.raise_for_status()
            res = post_response.json()
            logging.info(f"Created new session successfully")
        except Exception as e:
            logging.warning(f"Failed to create new session: {e}")
            # Return empty response to trigger error handling
            raise Exception(f"No existing session and failed to create new one: {e}")
    
    # Handle response (could be array or single object)
    is_arr = isinstance(res, list) and len(res) > 0
    if is_arr:
        res = res[0]
    
    session_response = TypeAdapter(ComputeApiSessionResponse).validate_python(res)
    
    # Use the real session data from compute API
    compute_config.session_endpoint = session_response.session_endpoint
    compute_config.session_uuid = session_response.session_uuid
    compute_config.access_token = session_response.access_token
    
    logging.info(f"✓ Fetched session from compute API:")
    logging.info(f"  - Session UUID: {session_response.session_uuid}")
    logging.info(f"  - Session endpoint: {session_response.session_endpoint}")
    logging.info(f"  - Access token received: {'Yes' if session_response.access_token else 'No'}")
    
    # Critical check: access_token must be present for authentication
    if not session_response.access_token or len(session_response.access_token.strip()) == 0:
        logging.error(f"CRITICAL: No access_token received from session response!")
        if isinstance(res, dict):
            logging.error(f"Session response keys: {list(res.keys())}")
        else:
            logging.error(f"Session response type: {type(res)}")
        logging.error(f"This means the M2M authentication is not working properly.")
        logging.error(f"The pipeline will fail to authenticate with dataset-api and model resolution will fail.")
        raise Exception("CRITICAL: No access_token received from compute API session response. M2M authentication is not configured properly.")
    
    logging.debug(f"✓ Received access_token from session response (length: {len(session_response.access_token)})")
    
    pipeline_id = session_response.pipelines[0]["pipeline_id"] if len(session_response.pipelines) > 0 else ""
    # if not pipeline_id:
    #     logging.error(f"Cannot get the pipeline id. Something went wrong: {res}")
    #     raise Exception("Cannot get the pipeline id. Something went wrong")
    compute_config.pipeline_id = pipeline_id
    return compute_config