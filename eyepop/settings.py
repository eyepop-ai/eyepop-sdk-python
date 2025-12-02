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
