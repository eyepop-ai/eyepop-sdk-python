
import logging
import time

import requests

from eyepop.compute.models import ComputeApiSessionResponse, ComputeContext, PipelineStatus

log = logging.getLogger('eyepop.compute')


class WaitException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def wait_for_session(compute_config: ComputeContext) -> bool:
    timeout = compute_config.wait_for_session_timeout
    interval = compute_config.wait_for_session_interval
    
    # Session endpoint health check ALWAYS uses the JWT access_token
    if not compute_config.access_token or len(compute_config.access_token.strip()) == 0:
        raise Exception(
            "No access_token in compute_config. "
            "Cannot perform session health check. "
            "This should never happen - fetch_new_compute_session should have set it."
        )

    auth_header = f"Bearer {compute_config.access_token}"
    log.debug(f"Using JWT access_token for session health check at {compute_config.session_endpoint}/health")

    headers = {
        "Authorization": auth_header,
        "Accept": "application/json",
    }
    
    health_url = f"{compute_config.session_endpoint}/health"
    log.debug(f"Waiting for session to be ready at: {health_url}")
    log.debug(f"Timeout: {timeout}s, Interval: {interval}s")
    
    end_time = time.time() + timeout
    last_message = "No message received"
    attempt = 0
    while time.time() < end_time:
        attempt += 1
        try:
            log.debug(f"Health check attempt {attempt}")
            response = requests.get(health_url, headers=headers)
            log.debug(f"Health check response status: {response.status_code}")
            
            if response.status_code == 200:
                log.info("Session is ready (status 200)")
                return True
            
            # If not 200, try to parse the response as session info
            if response.status_code != 200:
                last_message = f"Health check returned status {response.status_code}"
                log.debug(last_message)
                time.sleep(interval)
                continue
                
            session_response = ComputeApiSessionResponse(**response.json())

            # Check status - the enum's _missing_ method will handle unknown values
            status = session_response.session_status
            
            if status == PipelineStatus.RUNNING:
                log.info("Session is running")
                return True
            elif status == PipelineStatus.PENDING:
                # Continue waiting for these statuses
                last_message = f"Session status: {status.value}"
                log.debug(f"Session still pending/creating: {last_message}")
                time.sleep(interval)
                continue
            elif status in [PipelineStatus.FAILED, PipelineStatus.ERROR, PipelineStatus.STOPPED]:
                # Terminal states - stop waiting
                raise WaitException(f"Session in terminal state: {status.value}. Message: {session_response.session_message}")
            else:
                # Unknown status - log and continue waiting
                last_message = f"Session status: {status.value}"
                log.debug(f"Unknown session status, continuing to wait: {last_message}")
                time.sleep(interval)
                continue

        except WaitException as e:
            last_message = str(e)
            log.warning(f"Session wait exception: {last_message}")
        except Exception as e:
            # Don't immediately raise, just log and continue trying
            last_message = str(e)
            log.debug(f"Exception during health check: {last_message}")
        time.sleep(interval)

    log.error(f"Session timed out after {timeout}s. Last message: {last_message}")
    raise TimeoutError(f"Session timedout. Last message: {last_message}")