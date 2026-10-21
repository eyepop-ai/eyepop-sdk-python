from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Self


@dataclass(frozen=True)
class _Inference:
    cropped: tuple["Ability | Model", ...] = field(default=(), kw_only=True)

    def crop(self, *targets: "Ability | Model") -> Self:
        """[EXPERIMENTAL] Also run `targets` on the crop of each object this finds."""
        return replace(self, cropped=self.cropped + targets)

    @property
    def json(self) -> dict[str, Any]:
        """A pop of this one component."""
        return {"components": [self._component()]}

    def _component(self) -> dict[str, Any]:
        component: dict[str, Any] = {"type": "inference", **self._reference()}
        if self.cropped:
            component["forward"] = {"operator": {"type": "crop"}, "targets": [t._component() for t in self.cropped]}
        return component

    def _reference(self) -> dict[str, str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Ability(_Inference):
    """[EXPERIMENTAL] An ability by name, such as `Ability("eyepop.person")`."""

    name: str

    def _reference(self) -> dict[str, str]:
        return {"ability": self.name}


@dataclass(frozen=True)
class Model(_Inference):
    """[EXPERIMENTAL] A model by its uuid."""

    uuid: str

    def _reference(self) -> dict[str, str]:
        return {"modelUuid": self.uuid}


@dataclass(frozen=True)
class Pop:
    """[EXPERIMENTAL] A pop's JSON, sent as written."""

    json: Mapping[str, Any]
