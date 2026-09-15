import torch
from loguru import logger

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Set seed to {seed}")

def resolve_device(device_arg: str | None) -> torch.device:
    try:
        if device_arg is not None:
            logger.info(f"Using device {device_arg}")
            return torch.device(device_arg)
    except Exception as e:
            logger.error(f"Invalid device string '{device_arg}': {e}")
            raise ValueError(f"Invalid device string '{device_arg}': {e}") from e

    logger.info("Using default device")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


