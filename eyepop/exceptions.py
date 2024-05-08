class PopNotStartedException(Exception):
    """
    Thrown when the given Pop was not started and the Endpoint was requested with auto_start=False
    """
    def __init__(self, pop_id: str):
        super().__init__(f'Pop {pop_id} has not been started')


class PopConfigurationException(Exception):
    """
    Thrown when the given Pop configuration was inconsistent 
    """
    def __init__(self, pop_id: str, reason: str):
        super().__init__(f'Pop {pop_id} configuration issue: {reason}')


class PopNotReachableException(Exception):
    """
    Thrown when no healthy endpoint is available
    """
    def __init__(self, pop_id: str, endpoints: list[dict]):
        super().__init__(f'No healthy endpoint for Pop {pop_id} tried: {endpoints}')

