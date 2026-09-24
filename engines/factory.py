
from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from data_modules.dataset import STL10Dataset, CIFAR10Dataset
from engines.interfaces.irunner import IInferencer, ITester, ITrainer
from engines.interfaces.ibuilder import ITrainerBuilder, ITesterBuilder
from engines.interfaces.icommon import Criterion, Encoder, MaskSampler, Predictor, TorchModule
from engines.registries import DATASET_REGISTRY, TRAINER_BUILDER_REGISTRY, TESTER_BUILDER_REGISTRY
from engines.spec import TrainerBuildSpec
from models.convnext.model import ConvNext
from models.common import JepaMaskSampler
from loguru import logger

from utils.common_utils import raise_and_log

# OPTIMIZER_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {
#     "adam": torch.optim.Adam,
#     "adamw": torch.optim.AdamW
# }

# CRITERION_REGISTRY: dict[str, type[torch.nn.Module]] = {
#     "smooth_l1": torch.nn.SmoothL1Loss,
#     "mse": torch.nn.MSELoss,
#     "l1": torch.nn.L1Loss,
# }

ENCODER_REGISTRY: dict[str, type[Encoder]] = {
    "convnext": ConvNext
}


class Factory:
    
    def build_dataset(self, data_cfg: dict[str, Any]) -> Dataset:
        if "name" not in data_cfg and "type" not in data_cfg:
            raise_and_log("Dataset must have a `name`/`type` field")
            
        dataset_name = data_cfg["name"].lower() if "name" in data_cfg else data_cfg["type"].lower()
        kwargs = {k: v for k, v in data_cfg.items() if k != "name"}
        dataset = DATASET_REGISTRY.build(dataset_name, **kwargs)
        logger.debug(f"Built dataset '{dataset_name}'")
        return dataset
        
    def build_dataloader(self, data_cfg: dict[str, Any], is_train: bool) -> DataLoader:
        dataloader = DataLoader(
            self.build_dataset(data_cfg),
            batch_size=data_cfg["batch_size"],
            num_workers=data_cfg.get("num_workers", 1),
            shuffle = is_train
        )
        logger.info(f"Built dataloader with {len(dataloader)} batches")
        return dataloader
    
    def build_trainer(self, cfg: dict[str, Any]) -> ITrainer:
        trainer_name = cfg["name"].lower()
        trainer_cls: ITrainerBuilder = TRAINER_BUILDER_REGISTRY.get(trainer_name)
        
        for c in trainer_cls.required_states():
            if c not in cfg:
                raise_and_log(f"Trainer builder {trainer_name} requires component {c}")
        
        
        trainer = trainer_cls(**trainer_cls.build_kwargs(cfg))
    
        if not isinstance(trainer, ITrainer):
            raise_and_log(f"Trainer {trainer_name} must implement ITrainer interface")
        
        return trainer
        
    def build_tester(self, cfg: dict[str, Any]) -> ITester:
        tester_name = cfg["name"].lower()
        tester_cls: ITesterBuilder = TESTER_BUILDER_REGISTRY.get(tester_name)

        for c in tester_cls.required_components():
            if c not in cfg:
                raise ValueError(f"Tester builder {tester_name} requires config field '{c}'")

        init_kwargs = tester_cls.build_kwargs(cfg)
        tester = tester_cls(**init_kwargs)

        if not isinstance(tester, ITester):
            raise TypeError(f"Tester {tester_name} must implement ITester")

        logger.info(f"Built tester {tester_name}")
        return tester
    
    def build_inferencer(self, cfg: dict[str, Any]) -> IInferencer:
        ...
    
    
# class IJepaFactory(BaseFactory):
    
#     def __init__(self) -> None:
#         self.expected_components = [
#             "encoder",
#             "predictor",
#             "mask_sampler",
#             "optimizer",
#             "criterion"
#         ]
        
#     def _initial_check(self, cfg: dict[str, Any]):
#         for c in self.expected_components:
#             if c not in cfg:
#                 logger.error(f"Missing component {c}")
#                 raise ValueError(f"Missing component {c}")
            
#     def _build_enc(self, enc_cfg: dict[str, Any]) -> Tuple[Encoder, Encoder]:
#         enc_name = enc_cfg["name"].lower()
        
#         if enc_name not in ENCODER_REGISTRY:
#             logger.error(f"Unknown encoder {enc_name}")
#             raise ValueError(f"Unknown encoder {enc_name}")
                
#         encoder = ENCODER_REGISTRY[enc_name]
        
#         enc_params = enc_cfg["params"]
        
#         context_encoder= encoder(**enc_params)
#         tgt_encoder= encoder(**enc_params)
#         # special to ema as ctx and tgt start out the same
#         #TODO: potentially hand over this responsibility at the resume check
#         tgt_encoder.load_state_dict(context_encoder.state_dict())
        
#         return context_encoder, tgt_encoder
    

#     def _build_predictor(self, predictor_cfg: dict[str, Any]) -> Predictor:
#         raise NotImplementedError("Predictors not yet implemented")
    
    
#     def _build_trainer_internal(self, cfg: dict[str, Any]) -> IJepaTrainer:
#         # for jepa trainer we expect the following structure
#         # context_encoder: {...}
#         # target_encoder: {...}
#         # predictor: {...}
#         # mask_sampler: {...}
#         # optimizer: {...}
#         # criterion: {...}
        
#         self._initial_check(cfg)
#         # Build components
#         context_encoder, tgt_encoder = self._build_enc(cfg["encoder"])
#         predictor = self._build_predictor(cfg["predictor"])
#         mask_sampler = JepaMaskSampler(**cfg["mask_sampler"])
#         optimizer = self.build_optimizer(cfg["optimizer"], [context_encoder, predictor])
#         criterion = self.build_criterion(cfg["criterion"])
        
#         return IJepaTrainer(
#             context_encoder=context_encoder,
#             target_encoder=tgt_encoder,
#             predictor=predictor,
#             mask_sampler=mask_sampler,
#             optimizer=optimizer,
#             criterion=criterion
#         )
    
#     def build_tester(self, cfg: dict[str, Any]) -> ITester:
#         raise NotImplementedError("Testers not yet implemented")
#     def build_inferencer(self, cfg: dict[str, Any]) -> IInferencer:
#         raise NotImplementedError("Inferencers not yet implemented")
    
    