import os
from typing import Union

from deprecated import deprecated
from matplotlib.axes import Axes

from eyepop.data.data_endpoint import DataEndpoint
from eyepop.data.data_syncify import SyncDataEndpoint
from eyepop.visualize import EyePopPlot
from eyepop.worker.worker_endpoint import WorkerEndpoint
from eyepop.worker.worker_syncify import SyncWorkerEndpoint


class EyePopSdk:
    """
    EyePop.ai Python SDK for Worker API
    """

    @staticmethod
    def workerEndpoint(pop_id: str | None = None, secret_key: str | None = None, auto_start: bool = True,
                       stop_jobs: bool = True, eyepop_url: str | None = None, job_queue_length: int = 1024,
                       is_async: bool = False, is_sandbox: bool = False,
                       request_tracer_max_buffer: int = 1204) -> WorkerEndpoint | SyncWorkerEndpoint:
        if secret_key is None:
            secret_key = os.getenv('EYEPOP_SECRET_KEY')
            if secret_key is None:
                raise KeyError('parameter \'secret_key\' or environment \'EYEPOP_SECRET_KEY\' is required')

        if eyepop_url is None:
            eyepop_url = os.getenv('EYEPOP_URL')
            if eyepop_url is None:
                eyepop_url = 'https://api.eyepop.ai'

        if pop_id is None:
            pop_id = os.getenv('EYEPOP_POP_ID')
            if pop_id is None:
                raise KeyError('parameter \'pop_id\' is required')

        endpoint = WorkerEndpoint(secret_key=secret_key, pop_id=pop_id, auto_start=auto_start, stop_jobs=stop_jobs,
                                  eyepop_url=eyepop_url, job_queue_length=job_queue_length, is_sandbox=is_sandbox,
                                  request_tracer_max_buffer=request_tracer_max_buffer)

        if not is_async:
            endpoint = SyncWorkerEndpoint(endpoint)

        return endpoint

    """
    EyePop.ai Python SDK for Data API
    """

    @staticmethod
    def dataEndpoint(account_id: str | None = None, secret_key: str | None = None, eyepop_url: str | None = None,
                     job_queue_length: int = 1024, is_async: bool = False,
                     request_tracer_max_buffer: int = 1204) -> Union[DataEndpoint, SyncDataEndpoint]:
        if secret_key is None:
            secret_key = os.getenv('EYEPOP_SECRET_KEY')
            if secret_key is None:
                raise KeyError('parameter \'secret_key\' or environment \'EYEPOP_SECRET_KEY\' is required')

        if eyepop_url is None:
            eyepop_url = os.getenv('EYEPOP_URL')
            if eyepop_url is None:
                eyepop_url = 'https://api.eyepop.ai'

        if account_id is None:
            account_id = os.getenv('EYEPOP_ACCOUNT_ID')
            if account_id is None:
                raise KeyError('parameter \'account_id\' is required')

        endpoint = DataEndpoint(secret_key=secret_key, account_id=account_id, eyepop_url=eyepop_url,
                                job_queue_length=job_queue_length, request_tracer_max_buffer=request_tracer_max_buffer)

        if not is_async:
            endpoint = SyncDataEndpoint(endpoint)

        return endpoint

    @staticmethod
    def plot(axes: Axes):
        return EyePopPlot(axes)

    @staticmethod
    @deprecated(version='0.19.0', reason="use workerEndpoint() instead, will be removed in v1.0.0")
    def endpoint(**kwargs):
        return EyePopSdk.workerEndpoint(**kwargs)
