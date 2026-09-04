from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from engines.interfaces.icommon import Encoder, Predictor, MaskSampler, Criterion, CallBack, TorchModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Tuple
from loguru import logger

class BaseTrainer(ABC):
    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.allowed_checkpoint_format: list[str] = [".pth", ".pkl", ".pt"]
        self.device: torch.device = torch.device("cpu") # default on cpu, will be set to gpu on trainer.to(device)
    
    def _load_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        # currently only support loading from .pth, .pkl or .pt | .onnx will be supported in the future
        file_type = Path(checkpoint_path).suffix
        if file_type not in self.allowed_checkpoint_format:
            logger.error(f"Unsupported checkpoint file type {file_type}")
            raise ValueError(f"Unsupported checkpoint file type {file_type}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path)
        self.load_state_dict(state)
        
    def fit(self, data_loader: DataLoader, num_epochs: int, output_path: str | Path, call_backs: list[CallBack] | None = None, resume_path: str | Path | None = None) -> None:
            resume_path = Path(resume_path) if resume_path is not None else None
            if resume_path is not None:
                if not resume_path.exists():
                    logger.error(f"Resume path {resume_path} does not exist")
                    raise ValueError(f"Resume path {resume_path} does not exist")
                self._load_from_checkpoint(resume_path)
            call_backs = call_backs or []
            try:
                for epoch in range(self.current_epoch, num_epochs):
                    for batch in tqdm(data_loader):
                        metrics = self.training_step(batch)
                        for cb in call_backs:
                            cb.on_step_end(metrics, self.global_step)
                        self.global_step += 1
                    self.current_epoch += 1
                    
                    for cb in call_backs:
                        cb.on_epoch_end(epoch)
            except Exception:
                logger.exception("Training failed at epoch={} step={}", self.current_epoch, self.global_step)
                raise
            finally:
                for cb in call_backs:
                    cb.on_training_end()
                
                
    @abstractmethod
    def training_step(self, x: torch.Tensor) -> dict[str, float]:
        ...
        
    def base_state(self) -> dict[str, Any]:
        return {"epoch": self.current_epoch, "step": self.global_step}
    
    @abstractmethod
    def _component_state_dict(self) -> dict[str, Any]:
        ...
    def state_dict(self) -> dict[str, Any]:
        state: dict[str,  Any] = dict(self.base_state())
        state.update(self._component_state_dict())
        return state

    @abstractmethod
    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        ...
    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.current_epoch = state.get("epoch", 0)
        self.global_step = state.get("step", 0)
        self._load_component_state_dict(state)
        
    @abstractmethod
    def _to_components(self, device: torch.device) -> None:
        ...
    def to(self, device: torch.device) -> "BaseTrainer":
        self.device = device
        self._to_components(device)
        return self
    
class IJepaTrainer(BaseTrainer):
    def __init__(self
                 , context_encoder: Encoder
                 , target_encoder: Encoder
                 , predictor: Predictor
                 , mask_sampler: MaskSampler
                 , optimizer: Optimizer
                 , criterion: Criterion
                 ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.mask_sampler = mask_sampler
        self.optimizer = optimizer
        self.criterion = criterion
        
        for p in target_encoder.parameters():
            p.requires_grad = False
        
        
    def training_step(self, x: torch.Tensor) -> dict[str, float]:
        
        ctx_idx, tgt_idx_list = self.mask_sampler.sample(x) 
        
        context_reprs = self.context_encoder(x, ctx_idx)
        
        with torch.no_grad():
            full_target_reprs = self.target_encoder(x)
            
            target_reprs_list = [self._gather(full_target_reprs, tgt_idx) for tgt_idx in tgt_idx_list]
        
        preds = [self.predictor(context_reprs, tgt_idx) for tgt_idx in tgt_idx_list]
        
        losses = [self.criterion(pred, target_repr) for pred, target_repr in zip(preds, target_reprs_list)]
        
        loss = torch.stack(losses).sum()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item()
            }
        
    def _get_components(self) -> dict[str, TorchModule]:
        return {
            "context_encoder": self.context_encoder,
            "target_encoder": self.target_encoder,
            "predictor": self.predictor
        }
        
    def _component_state_dict(self) -> dict[str, Any]:
        state = {k: v.state_dict() for k, v in self._get_components().items()}
        state["optimizer"] = self.optimizer.state_dict()
        return state
    
    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        for k, v in self._get_components().items():
            v.load_state_dict(state[k])
        self.optimizer.load_state_dict(state["optimizer"])
        
    def _to_components(self, device: torch.device) -> None:
        self.context_encoder = self.context_encoder.to(device)
        self.target_encoder = self.target_encoder.to(device)
        self.predictor = self.predictor.to(device)
    
    @staticmethod
    def _gather(reprs: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(reprs, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, reprs.size(-1)))
        
    
    