import torch

from engines.interfaces.ibuilder import ITrainerBuilder, ITesterBuilder
from engines.interfaces.icommon import Criterion
from engines.interfaces.irunner import CallBack
from utils.common_utils import Registry
from torch.utils.data import Dataset

TRAINER_BUILDER_REGISTRY: Registry[ITrainerBuilder] = Registry("trainer_builder")
CALLBACK_REGISTRY: Registry[CallBack] = Registry("callback")
DATASET_REGISTRY: Registry[Dataset] = Registry("dataset")
TESTER_BUILDER_REGISTRY: Registry[ITesterBuilder] = Registry("tester_builder")
CRITERION_REGISTRY: Registry[Criterion] = Registry("criterion")

# common criterions
CRITERION_REGISTRY.register("smooth_l1")(torch.nn.SmoothL1Loss)
CRITERION_REGISTRY.register("mse")(torch.nn.MSELoss)
CRITERION_REGISTRY.register("l1")(torch.nn.L1Loss)
CRITERION_REGISTRY.register("cross_entropy")(torch.nn.CrossEntropyLoss)

