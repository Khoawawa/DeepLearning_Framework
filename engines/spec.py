from dataclasses import dataclass
from typing import Any

from engines.interfaces.icommon import TorchModule


@dataclass
class OptimizerSpec:
    kwarg_name: str
    cfg_key: str
    modules: list[TorchModule]

@dataclass
class TrainerBuildSpec:
    unique_kwargs: dict[str, Any]
    optimizer_specs: list[OptimizerSpec]