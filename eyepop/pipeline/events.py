from dataclasses import dataclass

from eyepop.pipeline.prediction import Prediction


@dataclass(frozen=True)
class Started:
    """[EXPERIMENTAL] The pipeline plays the source."""

    source: str


@dataclass(frozen=True)
class Dropped:
    """[EXPERIMENTAL] A live source: the server dropped `count` predictions because the client was slow."""

    source: str
    count: int


@dataclass(frozen=True)
class End:
    """[EXPERIMENTAL] A source ended: `reason` is `eos`, `error` or `preempted`."""

    source: str
    reason: str
    message: str = ""


@dataclass(frozen=True)
class Error:
    """[EXPERIMENTAL] A command failed. `source` is None for a command that is not a source; the socket stays open."""

    source: str | None
    error: str
    message: str


@dataclass(frozen=True)
class Done:
    """[EXPERIMENTAL] The socket has no open source left. It stays open for the next one."""


Event = Prediction | Started | Dropped | End | Error | Done


class _Stop:
    def __repr__(self) -> str:
        return "STOP"


STOP = _Stop()
"""[EXPERIMENTAL] Return it from a handler to stop. The contract has no command to stop one source, so it ends the pipeline."""
