from typing import Any, Protocol, runtime_checkable

from engines.spec import TesterBuildSpec, TrainerBuildSpec


@runtime_checkable
class ITrainerBuilder(Protocol):
    @classmethod
    def required_components(cls) -> list[str]: ...
    @classmethod
    def build_unique_kwargs(cls, cfg: dict[str, Any]) -> TrainerBuildSpec: ...

class ITesterBuilder(Protocol):
    @classmethod
    def required_components(cls) -> list[str]: ...
    @classmethod
    def build_unique_kwargs(cls, cfg: dict[str, Any]) -> TesterBuildSpec: ...