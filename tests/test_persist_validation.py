import pytest

from eyepop import EyePopSdk
from eyepop.exceptions import PopConfigurationException


def test_persist_without_pop_id_raises():
    with pytest.raises(PopConfigurationException, match="persist requires pop_id"):
        EyePopSdk.sync_worker(
            pop_id="transient",
            api_key="test-key",
            persist=True,
        )


def test_pop_id_and_session_uuid_raises():
    with pytest.raises(PopConfigurationException, match="cannot pass both"):
        EyePopSdk.sync_worker(
            pop_id="abc-123",
            session_uuid="xyz-789",
            secret_key="test-key",
        )


def test_persist_false_does_not_raise():
    """persist=False (default) should not trigger validation errors."""
    # This will fail later when trying to connect, but should not fail on validation
    endpoint = EyePopSdk.async_worker(
        pop_id="transient",
        api_key="test-key",
        persist=False,
    )
    assert endpoint is not None
    assert not endpoint._persist
