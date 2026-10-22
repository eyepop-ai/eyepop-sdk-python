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



class PipelineSocketClosedException(Exception):
    """Thrown for a job in flight when the pipeline API's socket closed; the job is not sent again.

    `code` and `reason` are the close's, for instance 1011 `pipeline_stopped`.
    """
    def __init__(self, code: int | None, reason: str):
        self.code = code
        self.reason = reason
        super().__init__(f'the pipeline API closed the socket: {code} {reason}'.rstrip())


class PipelineCommandException(Exception):
    """Thrown when the pipeline API answered a command with an error, for instance `queue_full`."""
    def __init__(self, error: str, message: str = ''):
        self.error = error
        self.message = message
        super().__init__(f'the pipeline API refused the command: {error} {message}'.rstrip())


class PipelineSourceException(ValueError):
    """Thrown when a source ended with reason `error` or `preempted` rather than at its end of stream."""
    def __init__(self, source_id: str, reason: str, message: str = ''):
        self.source_id = source_id
        self.reason = reason
        self.message = message
        super().__init__(f'Error in source {source_id}: {reason} {message}'.rstrip())
