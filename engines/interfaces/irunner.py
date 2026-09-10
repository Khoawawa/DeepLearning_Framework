from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
import torch
from torch.utils.data import DataLoader




@runtime_checkable
class ITrainer(Protocol):
    
    def fit(self, data_loader: DataLoader, num_epochs: int, call_backs: list["CallBack"] | None = None, resume_path: str | Path | None = None) -> None:
        ...
    
    def training_step(self, x: torch.Tensor ) -> dict[str, float]:
        ...
    
    def state_dict(self) -> dict[str, Any]:
        ...
    
    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        ...
    
    def to(self, device: torch.device) -> "ITrainer":
        ...
@runtime_checkable
class CallBack(Protocol):

    def on_step_end(self, trainer: ITrainer, metrics: dict[str, float], step: int) -> None:
        ...

    def on_epoch_end(self, trainer: ITrainer, epoch: int) -> None:
        ...

    def on_training_end(self, trainer: ITrainer) -> None:
        ...
@runtime_checkable
class IInferencer(Protocol):
    def infer(self, x:torch.Tensor) -> torch.Tensor:
        ...
    def to(self, device: torch.device) -> "IInferencer":
        ... 
@runtime_checkable
class ITester(Protocol):
    def test(self, data_loader: DataLoader) -> None:
        ...
    def to(self, device: torch.device) -> "ITester":
        ...