from typing import Protocol, runtime_checkable, Any

import torch
from torch.utils.data import DataLoader, Dataset

from engines.interfaces.icommon import TorchModule
from engines.interfaces.irunner import IInferencer, ITester, ITrainer

@runtime_checkable
class IFactory(Protocol):
    
    def build_trainer(self, cfg: dict[str, Any]) -> ITrainer:
        ...
    def build_tester(self, cfg: dict[str, Any]) -> ITester: ...
    
    def build_inferencer(self, cfg: dict[str, Any]) -> IInferencer: ...
    
    def build_dataloader(self, data_cfg: dict[str, Any], is_train: bool) -> DataLoader:
        ...
    
    def build_dataset(self, data_cfg: dict[str, Any]) -> Dataset:
        ...