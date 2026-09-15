from typing import Any, Protocol, runtime_checkable

from engines.spec import TrainerBuildSpec


@runtime_checkable
class ITrainerBuilder(Protocol):
    @classmethod
    def required_states(cls) -> list[str]: ...
    @classmethod
    def build_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]: ...