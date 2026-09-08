
from typing import Any

from torch.autograd import Variable
from loguru import logger
import numpy as np
import torch

from engines.factory import IJepaFactory
from engines.interfaces.ifactory import IFactory


FACTORY_REGISTRY: dict[str, type[IFactory]] = {
    "i_jepa": IJepaFactory
}

def resolve_factory(trainer_name: str) -> IFactory:
    if trainer_name == "":
        logger.error("Factory not specified")
        raise ValueError("Factory not specified")
    if trainer_name not in FACTORY_REGISTRY:
        logger.error(f"Unsupported factory producing {trainer_name}, supported factory: {list(FACTORY_REGISTRY.keys())}")
        raise ValueError(f"Unsupported factory producing {trainer_name}, supported factory: {list(FACTORY_REGISTRY.keys())}")
    
    logger.info(f"Using factory of {trainer_name}")
    return FACTORY_REGISTRY[trainer_name]()


def to_var(d: Any, device: torch.device) -> Any:
    if torch.is_tensor(d):
        data = Variable(d)
        if torch.cuda.is_available():
            data = data.to(device)
        return data
    if isinstance(d, np.ndarray):
        d_tensor = torch.from_numpy(d)
        return to_var(d_tensor, device)
    if isinstance(d, int) or isinstance(d, float):
        return d
    if isinstance(d, dict):
        for k in d.keys():
            d[k] = to_var(d[k], device)
        return d
    if isinstance(d, list):
        return list(map(lambda x: to_var(x, device), d))
    
    logger.error(f"to_var util doesnt support to operation of type: {type(d)}")
    raise ValueError(f"to_var util doesnt support to operation of type: {type(d)}")
