import os
import warnings

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='EYEPOP_')
    log_level: str = "INFO"
    session_timeout: int = 60
    session_interval: int = 2
    default_compute_url: str = "https://compute.eyepop.ai"
    min_config_reconnect_secs: float = 10.0
    max_retry_time_secs: float = 30.0
    force_refresh_config_secs: float = 3721.0  # 61 * 61
    send_trace_threshold_secs: float = 10.0
    default_job_queue_length: int = 1024
    default_request_tracer_max_buffer: int = 1204
    ws_initial_reconnect_delay: float = 1.0
    ws_max_reconnect_delay: float = 60.0
    confidence_n_digits: int = 3
    coordinate_n_digits: int = 3
    embedding_n_digits: int = 1
    default_data_url: str = "https://dataset-api.eyepop.ai"


settings = Settings()


def account_uuid_from_env(stacklevel: int = 3) -> str | None:
    """The account named by the environment.

    Reads EYEPOP_ACCOUNT_UUID, the name every EyePop tool uses, then the deprecated EYEPOP_ACCOUNT_ID.
    """
    account_uuid = os.getenv("EYEPOP_ACCOUNT_UUID")
    if account_uuid is None:
        account_uuid = os.getenv("EYEPOP_ACCOUNT_ID")
        if account_uuid is not None:
            warnings.warn("EYEPOP_ACCOUNT_ID is deprecated, use EYEPOP_ACCOUNT_UUID instead",
                          DeprecationWarning, stacklevel=stacklevel)
    return account_uuid
