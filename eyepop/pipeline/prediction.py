from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictedObject:
    """[EXPERIMENTAL] One object a pipeline found. `raw` holds it as the pipeline wrote it, key points and texts included."""

    label: str | None
    confidence: float | None
    x: float
    y: float
    width: float
    height: float
    objects: list["PredictedObject"]
    raw: Mapping[str, Any]

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> "PredictedObject":
        return cls(
            data.get("classLabel"),
            data.get("confidence"),
            float(data.get("x", 0)),
            float(data.get("y", 0)),
            float(data.get("width", 0)),
            float(data.get("height", 0)),
            [cls.parse(o) for o in data.get("objects") or []],
            data,
        )


@dataclass(frozen=True)
class Prediction:
    """[EXPERIMENTAL] One image's or one frame's prediction. `timestamp` is in nanoseconds from the media's start.

    `source` is the id of the source it came from on a socket, and None for an image sent alone.
    """

    timestamp: int | None
    source_width: float
    source_height: float
    objects: list[PredictedObject]
    raw: Mapping[str, Any]
    source: str | None = None

    @classmethod
    def parse(cls, data: Mapping[str, Any]) -> "Prediction":
        return cls(
            data.get("timestamp"),
            float(data.get("source_width", 0)),
            float(data.get("source_height", 0)),
            [PredictedObject.parse(o) for o in data.get("objects") or []],
            data,
        )


@dataclass(frozen=True)
class Result:
    """[EXPERIMENTAL] One prediction of a batch, or the `error` of an item that failed, with the `source` it came from."""

    source: Any
    prediction: Prediction | None = None
    error: str | None = None
