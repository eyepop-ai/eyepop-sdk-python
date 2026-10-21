class EyePopError(Exception):
    """[EXPERIMENTAL] Any field that did not arrive is None (`message` is "")."""

    def __init__(
        self,
        message: str = "",
        *,
        status: int | None = None,
        close_code: int | None = None,
        error: str | None = None,
    ):
        self.status = status
        self.close_code = close_code
        self.error = error
        self.message = message
        parts = [
            f"HTTP {status}" if status is not None else "",
            f"close {close_code}" if close_code is not None else "",
            error or "",
        ]
        head = " ".join(p for p in parts if p)
        super().__init__(f"{head}: {message}" if head and message else head or message)
