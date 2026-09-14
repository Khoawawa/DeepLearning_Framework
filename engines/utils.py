
from typing import Any

from torch.autograd import Variable
from loguru import logger
import numpy as np
import torch

from engines.interfaces.ifactory import IFactory


def resolve_factory(factory_name: str | None = None) -> IFactory:
    from engines.factory import Factory

    if factory_name is None:
        logger.info("Using default general factory")
        return Factory()

    # TODO: if there are more factory down the line replace this
    return Factory()



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
