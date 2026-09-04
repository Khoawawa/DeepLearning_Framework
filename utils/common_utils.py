from pathlib import Path
from typing import Any

import yaml
from loguru import logger


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    
    if not path.exists():
        logger.error(f"File {path} does not exist")
        raise FileNotFoundError(f"File {path} does not exist")
    with open(path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    
    if config is None:
        logger.error(f"Failed to load config from {path}")
        raise ValueError(f"Failed to load config from {path}")
    
    logger.info(f"Loaded config from {path}")
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    # for now it is load yaml but for the future this function is robust for hydra or other cofiguration frameworks too
    base_config = load_yaml(path)
    # 
    modified_config: dict[str, Any] = {}
    for k, v in base_config.items():
        if k == "defaults":
            for k2, v2 in v.items():
                modified_config[k2] = load_yaml(resolve_config_path(f"{k2}/{v2}"))
        else:
            modified_config[k] = v
    logger.info(f"Loaded modified config from {path}")
    return modified_config


def resolve_config_path(config_name: str) -> Path:
    p = Path(config_name)
    
    if p.suffix and p.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config name '{config_name}' has unexpected extension '{p.suffix}', expected .yaml/.yml")
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if p.is_absolute() or p.exists():
        logger.info(f"Config path {p} is absolute or exists")
        return p
    
    real_path = CONFIG_ROOT / p
    
    if not real_path.exists():
        logger.error(f"Config path {real_path} does not exist")
        raise FileNotFoundError(f"Config path {real_path} does not exist")
    
    logger.info(f"Resolving config path {p} to {real_path}")
        
    return real_path


def resolve_output_path(output_dir: str, run_name: str) -> Path:
    if (run_name := run_name.strip()) == "":
        logger.error("Run name cannot be empty")
        raise ValueError("Run name cannot be empty")
    
    output_path = CONFIG_ROOT / output_dir / run_name
    logger.info(f"Set output path to be {output_path}")
    
    return output_path
    