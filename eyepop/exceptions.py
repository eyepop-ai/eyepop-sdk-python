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


class PipelineOverloadedException(PipelineCommandException):
    """Thrown when the pipeline API had no capacity for what the caller asked.

    That is a live source refused `overloaded` twice, or a socket refused (HTTP 503 or 429) for
    up to `pipeline_wait_timeout` seconds. `waited_secs` is how long the SDK waited for a socket before it
    gave up; it is None for a source.
    """
    def __init__(self, message: str, waited_secs: float | None = None):
        # The message is the whole text; the parent would prefix it with 'refused the command'.
        Exception.__init__(self, message)
        self.error = 'overloaded'
        self.message = message
        self.waited_secs = waited_secs


class PipelineSourceException(ValueError):
    """Thrown when a source ended with reason `error` or `preempted` rather than at its end of stream."""
    def __init__(self, source_id: str, reason: str, message: str = ''):
        self.source_id = source_id
        self.reason = reason
        self.message = message
        super().__init__(f'Error in source {source_id}: {reason} {message}'.rstrip())


class MediaReadException(Exception):
    """Thrown when a read of a recording by its mediaId was refused (eyepop-instance spec 025).

    `status` is the HTTP status; `error`, `reason` and `message` are the answer's JSON body, `reason`
    being the code to act on: `invalid_media_id`, `invalid_request`, `invalid_bbox` or `clip_too_long`
    (400), `no_account` (403), `stream_not_found` (404, also for a mediaId of another session or account),
    `media_expired` or `media_gap` (410), `media_not_yet_available` (425, raised once the SDK's own retries
    after `Retry-After` ran out of time), `media_cache_not_configured`, `media_cache_scope_unresolved` or
    `overloaded` (503, raised once the SDK's retries after `Retry-After` ran out of time). `retry_after` is the answer's `Retry-After` in seconds, when it has one.
    """
    def __init__(self, status: int, error: str | None = None, reason: str | None = None, message: str = '',
                 retry_after: float | None = None):
        self.status = status
        self.error = error
        self.reason = reason
        self.message = message
        self.retry_after = retry_after
        detail = ' '.join(part for part in (reason or error, message) if part)
        super().__init__(f'the media read was refused: HTTP {status} {detail}'.rstrip())
