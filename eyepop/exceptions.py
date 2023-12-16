class PopNotStartedException(Exception):
    def __init__(self, pop_id: str):
        super().__init__(f'Pop {pop_id} has not been started')

