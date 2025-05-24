import os
import time

import requests

from eyepop.compute.models import ComputeApiSessionResponse, PipelineStatus


class WaitException(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


def wait_for_session(worker_url: str, interval: int = 1, timeout: int = 10) -> bool:
    headers = {
        "Authorization": f"Bearer {os.getenv('EYEPOP_SECRET_KEY')}",
        "Accept": "application/json",
    }
    end_time = time.time() + timeout
    last_message = "No message received"
    while time.time() < end_time:
        try:
            response = requests.get(f"{worker_url}/health", headers=headers)
            if response.status_code == 200:
                return True
            
            session_response = ComputeApiSessionResponse(**response.json())

            if session_response.session_status != PipelineStatus.RUNNING:
                raise WaitException(session_response.session_message)
            return True

        except WaitException as e:
            last_message = str(e)
        except Exception as e:
            raise e
        time.sleep(interval)

    raise TimeoutError(f"Session timedout. Last message: {last_message}")