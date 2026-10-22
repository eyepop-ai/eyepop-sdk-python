from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

import aiohttp

from eyepop.worker.worker_types import Pop

if TYPE_CHECKING:
    from eyepop.worker.worker_jobs import WorkerJob, WorkerSource


class WorkerBackend(ABC):
    """What runs a WorkerEndpoint's jobs: a worker session, or the pipeline API's socket.

    The endpoint owns the public methods, the credential, the HTTP client and
    the job queue; it describes each job as a WorkerSource and hands it to its
    backend to run.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Reach whatever runs jobs, when the endpoint connects."""

    @abstractmethod
    async def disconnect(self, timeout: float | None = None) -> None:
        """Release it, when the endpoint disconnects."""

    @abstractmethod
    async def set_pop(self, pop: Pop) -> Any:
        """Make `pop` the one that runs; the endpoint's pop is updated by the backend once it is."""

    @abstractmethod
    async def session(self) -> dict | None:
        """What a client needs to reach what runs the jobs."""

    def accept(self, source: "WorkerSource") -> None:  # noqa: B027
        """Check a job's source when it is submitted, before it is queued."""

    @abstractmethod
    async def run(self, job: "WorkerJob") -> None:
        """Run the job's source to its end, pushing each prediction to the job."""

    async def cancelled(self, job: "WorkerJob") -> None:  # noqa: B027
        """The job was cancelled; stop delivering its predictions."""

    # Worker session routes, which only a session backend has.

    async def dev_mode_base_url(self) -> str:
        raise _no_session('dev_mode_base_url')

    async def dev_mode_pipeline_base_url(self) -> str:
        raise _no_session('dev_mode_pipeline_base_url')

    async def pipeline_get(self, url_path_and_query: str, accept: str | None = None,
                           timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_get')

    async def pipeline_post(self, url_path_and_query: str, accept: str | None = None, open_data: Callable | None = None,
                            content_type: str | None = None,
                            timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_post')

    async def pipeline_patch(self, url_path_and_query: str, accept: str | None = None, data: Any = None,
                             content_type: str | None = None,
                             timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_patch')

    async def pipeline_delete(self, url_path_and_query: str,
                              timeout: aiohttp.ClientTimeout | None = None) -> aiohttp.ClientResponse:
        raise _no_session('pipeline_delete')


def _no_session(method: str) -> NotImplementedError:
    return NotImplementedError(f'{method} needs a worker session; pipeline mode has none')
