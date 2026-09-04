
from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from data_modules.dataset import STL10Dataset
from engines.interfaces.irunner import IInferencer, ITester, ITrainer
from engines.interfaces.icommon import Criterion, Encoder, MaskSampler, Predictor, TorchModule
from engines.trainer import IJepaTrainer
from models.convnext.model import ConvNext
from models.common import JepaMaskSampler
from loguru import logger

OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}

CRITERION_REGISTRY: dict[str, type[torch.nn.Module]] = {
    "smooth_l1": torch.nn.SmoothL1Loss,
    "mse": torch.nn.MSELoss,
    "l1": torch.nn.L1Loss,
}

ENCODER_REGISTRY: dict[str, type[Encoder]] = {
    "convnext": ConvNext
}

DATASET_REGISTRY: dict[str, type[Dataset]] = {
    "stl10": STL10Dataset
}

class BaseFactory:
    def __init__(self) -> None:
        pass
    
    def build_optimizer(self, optimizer_cfg: dict[str, Any], module_list: list[TorchModule]) -> torch.optim.Optimizer:
        optimizer_name = optimizer_cfg["name"].lower()
        if optimizer_name not in OPTIMIZER_REGISTRY:
            logger.error(f"Unknown optimizer {optimizer_name}")
            raise ValueError(f"Unknown optimizer {optimizer_name}")
        
        optimizer_cls = OPTIMIZER_REGISTRY[optimizer_name]
        params = [p for module in module_list for p in module.parameters()]
        kwargs = {k: v for k, v in optimizer_cfg.items() if k != "name"}
        try:
            built_optimizer = optimizer_cls(params, **kwargs)
        except Exception as e:
            logger.error(f"Failed to build optimizer '{optimizer_name}': {e}")
            raise
        return built_optimizer
    
    def build_criterion(self, criterion_cfg: dict[str, Any]) -> Criterion:
        criterion_name = criterion_cfg["name"].lower()
        if criterion_name not in CRITERION_REGISTRY:
            logger.error(f"Unknown criterion {criterion_name}")
            raise ValueError(f"Unknown criterion {criterion_name}")
        
        criterion_cls = CRITERION_REGISTRY[criterion_name]
        kwargs = {k: v for k, v in criterion_cfg.items() if k != "name"}
        try:
            built_criterion = criterion_cls(**kwargs)
        except Exception as e:
            logger.error(f"Failed to build criterion '{criterion_name}': {e}")
            raise
        
        return built_criterion
    
    def build_dataset(self, data_cfg: dict[str, Any]) -> Dataset:
        dataset_name = data_cfg["name"].lower()
        if dataset_name not in DATASET_REGISTRY:
            logger.error(f"Unknown dataset {dataset_name}")
            raise ValueError(f"Unknown dataset {dataset_name}")
        
        dataset_cls = DATASET_REGISTRY[dataset_name]
        kwargs = {k: v for k, v in data_cfg.items() if k != "name"}
        try:
            built_dataset = dataset_cls(**kwargs)
        except Exception as e:
            logger.error(f"Failed to build dataset '{dataset_name}': {e}")
            raise
        logger.info(f"Built dataset '{dataset_name}'")
        return built_dataset
        
    def build_dataloader(self, data_cfg: dict[str, Any], is_train: bool) -> DataLoader:
        dataloader = DataLoader(
            self.build_dataset(data_cfg),
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            shuffle = is_train
        )
        logger.info(f"Built dataloader with {len(dataloader)} batches")
        return dataloader
    
    @abstractmethod
    def build_trainer(self, cfg: dict[str, Any]) -> ITrainer:
        ...
    @abstractmethod
    def build_tester(self, cfg: dict[str, Any]) -> ITester:
        ...
    @abstractmethod
    def build_inferencer(self, cfg: dict[str, Any]) -> IInferencer:
        ...
    
    
class IJepaFactory(BaseFactory):
    
    def __init__(self) -> None:
        self.expected_components = [
            "encoder",
            "predictor",
            "mask_sampler",
            "optimizer",
            "criterion"
        ]
        
    def _initial_check(self, cfg: dict[str, Any]):
        for c in self.expected_components:
            if c not in cfg:
                logger.error(f"Missing component {c}")
                raise ValueError(f"Missing component {c}")
            
    def _build_enc(self, enc_cfg: dict[str, Any]) -> Tuple[Encoder, Encoder]:
        enc_name = enc_cfg["name"].lower()
        
        if enc_name not in ENCODER_REGISTRY:
            logger.error(f"Unknown encoder {enc_name}")
            raise ValueError(f"Unknown encoder {enc_name}")
                
        encoder = ENCODER_REGISTRY[enc_name]
        
        enc_params = enc_cfg["params"]
        
        context_encoder= encoder(**enc_params)
        tgt_encoder= encoder(**enc_params)
        # special to ema as ctx and tgt start out the same
        #TODO: potentially hand over this responsibility at the resume check
        tgt_encoder.load_state_dict(context_encoder.state_dict())
        
        return context_encoder, tgt_encoder
    

    def _build_predictor(self, predictor_cfg: dict[str, Any]) -> Predictor:
        raise NotImplementedError("Predictors not yet implemented")
    
    
    def build_trainer(self, cfg: dict[str, Any]) -> IJepaTrainer:
        # for jepa trainer we expect the following structure
        # context_encoder: {...}
        # target_encoder: {...}
        # predictor: {...}
        # mask_sampler: {...}
        # optimizer: {...}
        # criterion: {...}
        
        self._initial_check(cfg)
        # Build components
        context_encoder, tgt_encoder = self._build_enc(cfg["encoder"])
        predictor = self._build_predictor(cfg["predictor"])
        mask_sampler = JepaMaskSampler(**cfg["mask_sampler"])
        optimizer = self.build_optimizer(cfg["optimizer"], [context_encoder, predictor])
        criterion = self.build_criterion(cfg["criterion"])
        
        return IJepaTrainer(
            context_encoder=context_encoder,
            target_encoder=tgt_encoder,
            predictor=predictor,
            mask_sampler=mask_sampler,
            optimizer=optimizer,
            criterion=criterion
        )
    
    def build_tester(self, cfg: dict[str, Any]) -> ITester:
        raise NotImplementedError("Testers not yet implemented")
    def build_inferencer(self, cfg: dict[str, Any]) -> IInferencer:
        raise NotImplementedError("Inferencers not yet implemented")
    
    