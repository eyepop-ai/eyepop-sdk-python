import os

from matplotlib.axes import Axes

from eyepop.endpoint import Endpoint
from eyepop.syncify import SyncEndpoint
from eyepop.visualize import EyePopPlot


class EyePopSdk:
    """
    EyePop.ai Python SDK
    """
    @staticmethod
    def endpoint(pop_id: str | None = None, secret_key: str | None = None, auto_start: bool = True,
                 stop_jobs: bool = True, eyepop_url: str | None = None, job_queue_length: int = 1024,
                 is_async: bool = False, is_sandbox: bool = False, request_tracer_max_buffer: int = 1204) \
            -> Endpoint | SyncEndpoint:
        if secret_key is None:
            secret_key = os.getenv('EYEPOP_SECRET_KEY')
            if secret_key is None:
                raise Exception('parameter \'secret_key\' or environment \'EYEPOP_SECRET_KEY\' is required')

        if eyepop_url is None:
            eyepop_url = os.getenv('EYEPOP_URL')
            if eyepop_url is None:
                eyepop_url = 'https://api.eyepop.ai'

        if pop_id is None:
            pop_id = os.getenv('EYEPOP_POP_ID')
            if pop_id is None:
                raise Exception('parameter \'pop_id\' is required')

        endpoint = Endpoint(secret_key=secret_key, pop_id=pop_id, auto_start=auto_start, stop_jobs=stop_jobs,
                            eyepop_url=eyepop_url, job_queue_length=job_queue_length, is_sandbox=is_sandbox,
                            request_tracer_max_buffer=request_tracer_max_buffer)

        if not is_async:
            endpoint = SyncEndpoint(endpoint)

        return endpoint

    @staticmethod
    def plot(axes: Axes):
        return EyePopPlot(axes)

