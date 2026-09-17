from engines.interfaces.ibuilder import ITrainerBuilder, ITesterBuilder
from engines.interfaces.irunner import CallBack
from utils.common_utils import Registry
from torch.utils.data import Dataset

TRAINER_BUILDER_REGISTRY: Registry[ITrainerBuilder] = Registry("trainer_builder")
CALLBACK_REGISTRY: Registry[CallBack] = Registry("callback")
DATASET_REGISTRY: Registry[Dataset] = Registry("dataset")
TESTER_BUILDER_REGISTRY: Registry[ITesterBuilder] = Registry("tester_builder")
MODEL_REGISTRY = Registry("model")