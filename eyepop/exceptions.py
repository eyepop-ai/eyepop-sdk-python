class PopNotStartedException(Exception):
    """
    Thrown when the given Pop was not started and the Endpoint was requestred with auto_start=False
    """
    def __init__(self, pop_id: str):
        super().__init__(f'Pop {pop_id} has not been started')

