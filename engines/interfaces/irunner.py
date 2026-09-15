from pathlib import Path
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING
import torch
from torch.utils.data import DataLoader




@runtime_checkable
class IRunner(Protocol):
    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        ...

    def to(self, device: torch.device) -> "IRunner":
        ...


@runtime_checkable
class ITrainer(IRunner, Protocol):
    def fit(self, data_loader: DataLoader, num_epochs: int, call_backs: list["CallBack"] | None = None, resume_path: str | Path | None = None) -> None:
        ...

    def training_step(self, x: torch.Tensor) -> dict[str, float]:
        ...


@runtime_checkable
class CallBack(Protocol):
    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        ...

    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        ...

    def on_training_end(self, runner: IRunner) -> None:
        ...


@runtime_checkable
class IInferencer(IRunner, Protocol):
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        ...


@runtime_checkable
class ITester(IRunner, Protocol):
    def test(self, data_loader: DataLoader) -> None:
        ...