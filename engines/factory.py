
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

from utils.validation import Validation

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
        Validation.require_keys(data_cfg, ["name"], context="dataset config")
        name = data_cfg["name"].lower()
        if name not in DATASET_REGISTRY:
            raise_and_log(f"Unknown dataset '{name}'. "
                          f"Available: {DATASET_REGISTRY.keys()}")
        kwargs = {k: v for k, v in data_cfg.items() if k != "name"}
        dataset = DATASET_REGISTRY.build(name, **kwargs)
        logger.debug(f"Built dataset '{name}'")
        return dataset
        
    def build_dataloader(self, data_cfg: dict[str, Any], is_train: bool) -> DataLoader:
        Validation.require_keys(data_cfg, ["batch_size"], context="dataloader config")
        batch_size = data_cfg["batch_size"]
        if batch_size <= 0:
            raise_and_log(f"batch_size must be > 0, got {batch_size}")

        dataset = self.build_dataset(data_cfg)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=data_cfg.get("num_workers", 1),
            shuffle=is_train,
        )
        logger.info(f"Built dataloader with {len(dataloader)} batches")
        return dataloader
    
    def build_trainer(self, cfg: dict[str, Any]) -> ITrainer:
        Validation.require_keys(cfg, ["name"], context="trainer config")
        name = cfg["name"].lower()

        trainer_cls = TRAINER_BUILDER_REGISTRY.get(name)
        if trainer_cls is None:
            raise_and_log(
                f"Unknown trainer '{name}'. "
                f"Available: {list(TRAINER_BUILDER_REGISTRY.keys())}"
            )

        # 1) validate required state keys
        for key in trainer_cls.required_states():
            if key not in cfg:
                raise_and_log(
                    f"Trainer '{name}' requires config field '{key}'"
                )

        # 2) build kwargs (builder tự validate nội bộ)
        try:
            kwargs = trainer_cls.build_kwargs(cfg)
        except (KeyError, TypeError, AttributeError) as e:
            raise_and_log(f"Failed to build kwargs for trainer '{name}': {e}")

        if not isinstance(kwargs, dict):
            raise_and_log(
                f"{trainer_cls.__name__}.build_kwargs must return dict, "
                f"got {type(kwargs).__name__}"
            )
        # 3) instantiate
        try:
            trainer = trainer_cls(**kwargs)
        except TypeError as e:
            raise_and_log(f"Invalid kwargs for trainer '{name}': {e}")

        # 4) interface check
        if not isinstance(trainer, ITrainer):
            raise_and_log(
                f"Trainer '{name}' must implement ITrainer interface"
            )

        logger.info(f"Built trainer '{name}'")
        return trainer
        
    def build_tester(self, cfg: dict[str, Any]) -> ITester:
        Validation.require_keys(cfg, ["name"], context="tester config")
        name = cfg["name"].lower()

        tester_cls = TESTER_BUILDER_REGISTRY.get(name)
        if tester_cls is None:
            raise_and_log(
                f"Unknown tester '{name}'. "
                f"Available: {list(TESTER_BUILDER_REGISTRY.keys())}"
            )

        for key in tester_cls.required_components():
            if key not in cfg:
                raise_and_log(
                    f"Tester '{name}' requires config field '{key}'"
                )

        try:
            kwargs = tester_cls.build_kwargs(cfg)
        except (KeyError, TypeError, AttributeError) as e:
            raise_and_log(f"Failed to build kwargs for tester '{name}': {e}")

        try:
            tester = tester_cls(**kwargs)
        except TypeError as e:
            raise_and_log(f"Invalid kwargs for tester '{name}': {e}")

        if not isinstance(tester, ITester):
            raise_and_log(f"Tester '{name}' must implement ITester")

        logger.info(f"Built tester '{name}'")
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
    
    