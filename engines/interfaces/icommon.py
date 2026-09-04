from typing import Mapping, Protocol, runtime_checkable, Iterator, Any
from abc import abstractmethod
import torch
from typing_extensions import Self

@runtime_checkable
class TorchModule(Protocol):
    training: bool

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any: ...
    def to(self, *args: Any, **kwargs: Any) -> Self: ...
    def train(self, mode: bool = True) -> Self: ...
    def eval(self) -> Self: ...
    
    
@runtime_checkable
class Encoder(TorchModule, Protocol):
    @abstractmethod
    def __call__(self, x: torch.Tensor, mask: (torch.Tensor | None) = None) -> torch.Tensor: ...
    
@runtime_checkable
class Predictor(TorchModule, Protocol):
    @abstractmethod
    def __call__(self, context_repr: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """"""
        ...

@runtime_checkable
class MaskSampler(Protocol):
    @abstractmethod
    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        ...
        
@runtime_checkable
class Criterion(Protocol):
    @abstractmethod
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...

@runtime_checkable
class CallBack(Protocol):
    @abstractmethod
    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        ...
        
    @abstractmethod
    def on_epoch_end(self, epoch: int) -> None:
        ...
        
    @abstractmethod
    def on_training_end(self) -> None:
        ...