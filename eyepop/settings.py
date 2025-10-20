from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Main SDK settings.

    All environment variables use the prefix: EYEPOP_

    Access settings via:
        from eyepop.settings import settings

        timeout = settings.session_timeout
        max_retry = settings.max_retry_time_secs
    """
    model_config = SettingsConfigDict(env_prefix='EYEPOP_')

    # Compute API settings
    session_timeout: int = 60
    """Compute API session timeout (seconds)"""

    session_interval: int = 2
    """Compute API session check interval (seconds)"""

    default_compute_url: str = "https://compute.staging.eyepop.xyz"
    """Default Compute API URL"""

    # Worker API settings
    min_config_reconnect_secs: float = 10.0
    """Minimum time (seconds) between config reconnection attempts"""

    max_retry_time_secs: float = 30.0
    """Maximum time (seconds) to retry pipeline requests before failing"""

    force_refresh_config_secs: float = 3721.0  # 61 * 61
    """Force config refresh after this many seconds (1 hour + 1 minute)"""

    # General endpoint settings
    send_trace_threshold_secs: float = 10.0
    """Send request traces after this many seconds"""

    default_job_queue_length: int = 1024
    """Default maximum number of concurrent jobs in queue"""

    default_request_tracer_max_buffer: int = 1204
    """Default maximum number of trace events to buffer"""

    # Data API settings
    ws_initial_reconnect_delay: float = 1.0
    """Initial WebSocket reconnection delay (seconds)"""

    ws_max_reconnect_delay: float = 60.0
    """Maximum WebSocket reconnection delay (seconds)"""

    confidence_n_digits: int = 3
    """Number of decimal places for confidence values"""

    coordinate_n_digits: int = 3
    """Number of decimal places for coordinate values"""

    embedding_n_digits: int = 1
    """Number of decimal places for embedding values"""


# Global settings instance
settings = Settings()
