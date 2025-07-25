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
    def workerEndpoint(
            pop_id: str | None = None,
            secret_key: str | None = None,
            access_token: str | None = None,
            auto_start: bool = True,
            stop_jobs: bool = True,
            eyepop_url: str | None = None,
            job_queue_length: int = 1024,
            is_async: bool = False,
            is_sandbox: bool = False,
            is_local_mode: bool | None = None,
            request_tracer_max_buffer: int = 1204,
            dataset_uuid: str | None = None
    ) -> WorkerEndpoint | SyncWorkerEndpoint:
        if is_local_mode is None:
            is_local_mode = os.getenv('EYEPOP_LOCAL_MODE')
            if is_local_mode is not None:
                if is_local_mode.lower() != 'true' and is_local_mode.lower() != 'yes':
                    is_local_mode = None
        elif not is_local_mode:
            is_local_mode = None

        if access_token is None and secret_key is None:
            if is_local_mode is None:
                secret_key = os.getenv('EYEPOP_SECRET_KEY')
                if secret_key is None:
                    raise KeyError('parameter \'secret_key\' or environment \'EYEPOP_SECRET_KEY\' '
                                   'or parameter \'access_token\' is required')

        if eyepop_url is None:
            if is_local_mode is not None:
                eyepop_url = 'http://127.0.0.1:8080/standalone'
            else:
                eyepop_url = os.getenv('EYEPOP_URL')
                if eyepop_url is None:
                    eyepop_url = 'https://api.eyepop.ai'

        if pop_id is None:
            pop_id = os.getenv('EYEPOP_POP_ID')
            if pop_id is None:
                pop_id= 'transient'

        endpoint = WorkerEndpoint(
            secret_key=secret_key,
            access_token=access_token,
            pop_id=pop_id,
            auto_start=auto_start,
            stop_jobs=stop_jobs,
            eyepop_url=eyepop_url,
            job_queue_length=job_queue_length,
            is_sandbox=is_sandbox,
            request_tracer_max_buffer=request_tracer_max_buffer,
            dataset_uuid=dataset_uuid,
        )

        if not is_async:
            endpoint = SyncWorkerEndpoint(endpoint)

        return endpoint

    """
    EyePop.ai Python SDK for Data API
    """

    @staticmethod
    def dataEndpoint(account_id: str | None = None, secret_key: str | None = None, access_token: str | None = None,
                     eyepop_url: str | None = None, job_queue_length: int = 1024, is_async: bool = False,
                     request_tracer_max_buffer: int = 1204, disable_ws: bool = True) -> DataEndpoint | SyncDataEndpoint:
        if access_token is None and secret_key is None:
            secret_key = os.getenv('EYEPOP_SECRET_KEY')
            if secret_key is None:
                raise KeyError('parameter \'secret_key\' or environment \'EYEPOP_SECRET_KEY\' '
                               'or parameter \'access_token\' is required')

        if eyepop_url is None:
            eyepop_url = os.getenv('EYEPOP_URL')
            if eyepop_url is None:
                eyepop_url = 'https://api.eyepop.ai'

        if account_id is None:
            account_id = os.getenv('EYEPOP_ACCOUNT_ID')
            if account_id is None:
                raise KeyError('parameter \'account_id\' is required')

        endpoint = DataEndpoint(secret_key=secret_key, access_token=access_token,
                                account_id=account_id, eyepop_url=eyepop_url,
                                job_queue_length=job_queue_length, request_tracer_max_buffer=request_tracer_max_buffer,
                                disable_ws=disable_ws)

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
