
from loguru import logger

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