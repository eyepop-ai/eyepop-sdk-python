import aiohttp
import pytest

from eyepop.compute.context import ComputeContext, PipelineStatus
from eyepop.compute.status import wait_for_session
from eyepop.exceptions import ComputeHealthCheckException

HEALTH_URL = "https://session.example.com/health"


def _config(**overrides) -> ComputeContext:
    defaults = dict(
        session_endpoint="https://session.example.com",
        m2m_access_token="jwt-123",
        wait_for_session_timeout=5,
        wait_for_session_interval=1,
    )
    return ComputeContext(**(defaults | overrides))


@pytest.mark.asyncio
async def test_returns_true_on_running_pipeline(aioresponses):
    aioresponses.get(HEALTH_URL, payload={
        "session_uuid": "s1",
        "session_endpoint": "https://session.example.com",
        "access_token": "tok",
        "session_status": "running",
    })

    async with aiohttp.ClientSession() as session:
        assert await wait_for_session(_config(), session) is True


@pytest.mark.asyncio
async def test_returns_true_on_simple_health_response(aioresponses):
    """Health endpoint can return {"message": "I'm fine"} without session fields."""
    aioresponses.get(HEALTH_URL, payload={"message": "I'm fine"})

    async with aiohttp.ClientSession() as session:
        assert await wait_for_session(_config(), session) is True


@pytest.mark.asyncio
async def test_raises_on_terminal_state(aioresponses):
    aioresponses.get(HEALTH_URL, payload={
        "session_uuid": "s1",
        "session_endpoint": "https://session.example.com",
        "access_token": "tok",
        "session_status": "failed",
        "session_message": "OOM killed",
    })

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeHealthCheckException, match="terminal state"):
            await wait_for_session(_config(), session)


@pytest.mark.asyncio
async def test_retries_on_pending_then_succeeds(aioresponses):
    aioresponses.get(HEALTH_URL, payload={
        "session_uuid": "s1",
        "session_endpoint": "https://session.example.com",
        "access_token": "tok",
        "session_status": "pending",
    })
    aioresponses.get(HEALTH_URL, payload={
        "session_uuid": "s1",
        "session_endpoint": "https://session.example.com",
        "access_token": "tok",
        "session_status": "running",
    })

    async with aiohttp.ClientSession() as session:
        assert await wait_for_session(_config(), session) is True


@pytest.mark.asyncio
async def test_retries_on_non_200_then_succeeds(aioresponses):
    aioresponses.get(HEALTH_URL, status=503)
    aioresponses.get(HEALTH_URL, payload={
        "session_uuid": "s1",
        "session_endpoint": "https://session.example.com",
        "access_token": "tok",
        "session_status": "running",
    })

    async with aiohttp.ClientSession() as session:
        assert await wait_for_session(_config(), session) is True


@pytest.mark.asyncio
async def test_times_out_when_never_ready(aioresponses):
    for _ in range(100):
        aioresponses.get(HEALTH_URL, status=503)

    async with aiohttp.ClientSession() as session:
        with pytest.raises(TimeoutError, match="timed out"):
            await wait_for_session(_config(wait_for_session_timeout=1), session)


@pytest.mark.asyncio
async def test_raises_without_access_token():
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeHealthCheckException, match="No access_token"):
            await wait_for_session(_config(m2m_access_token=""), session)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", [
    PipelineStatus.FAILED,
    PipelineStatus.ERROR,
    PipelineStatus.STOPPED,
])
async def test_raises_on_all_terminal_states(aioresponses, terminal_status):
    aioresponses.get(HEALTH_URL, payload={
        "session_uuid": "s1",
        "session_endpoint": "https://session.example.com",
        "access_token": "tok",
        "session_status": terminal_status.value,
        "session_message": "bad",
    })

    async with aiohttp.ClientSession() as session:
        with pytest.raises(ComputeHealthCheckException, match="terminal state"):
            await wait_for_session(_config(), session)
