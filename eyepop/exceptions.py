class PopNotStartedException(Exception):
    """Thrown when the given Pop was not started and the Endpoint was requested with auto_start=False."""
    def __init__(self, pop_id: str):
        super().__init__(f'Pop {pop_id} has not been started')


class PopConfigurationException(Exception):
    """Thrown when the given Pop configuration was inconsistent."""
    def __init__(self, pop_id: str, reason: str):
        super().__init__(f'Pop {pop_id} configuration issue: {reason}')


class PopNotReachableException(Exception):
    """Thrown when no healthy endpoint is available."""
    def __init__(self, pop_id: str, endpoints: list[dict]):
        super().__init__(f'No healthy endpoint for Pop {pop_id} tried: {endpoints}')


class ComputeSessionException(Exception):
    """Thrown when compute API session creation or management fails."""
    def __init__(self, message: str, session_uuid: str | None = None):
        self.session_uuid = session_uuid
        super().__init__(message)


class ComputeTokenException(Exception):
    """Thrown when compute API token operations fail."""
    def __init__(self, message: str, session_uuid: str | None = None):
        self.session_uuid = session_uuid
        super().__init__(message)


class ComputeHealthCheckException(Exception):
    """Thrown when compute API health check fails or times out."""
    def __init__(self, message: str, session_endpoint: str | None = None, last_status: str | None = None):
        self.session_endpoint = session_endpoint
        self.last_status = last_status
        super().__init__(message)

