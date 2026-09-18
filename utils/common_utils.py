from pathlib import Path
from typing import Any, Callable, Generic, NoReturn, TypeVar

from dotenv import load_dotenv
import yaml
from loguru import logger

import inspect
from functools import wraps

CONFIG_ROOT = Path(__file__).resolve().parent.parent / "configs"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

def load_env() -> None:
    """Load .env into os.environ. override=False means anything already
    set in the environment (e.g. by launch.json's `env` block, or a real
    shell export) takes precedence over the file — .env is the fallback
    default, not an override."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)
    

def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise_and_log(f"File {path} does not exist", FileNotFoundError)
    with open(path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if config is None:
        raise_and_log(f"Failed to load config from {path}")

    logger.debug(f"Loaded config from {path}")
    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`. Nested dicts are merged
    key-by-key; any other type in `override` replaces the value in `base`
    outright (lists are replaced, not concatenated)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Hydra-style config loading: a `defaults:` block names config groups
    (e.g. `model: cnn` -> configs/model/cnn.yaml), each loaded and placed
    under its own top-level key (cfg["model"], cfg["data"], ...). Anything
    else in the file is merged on top at the true top level."""
    raw_config = load_yaml(path)
    defaults = raw_config.pop("defaults", {})

    merged: dict[str, Any] = {}
    for group, name in defaults.items():
        group_cfg = load_yaml(resolve_config_path(f"{group}/{name}"))
        merged[group] = deep_merge(merged.get(group, {}), group_cfg)

    merged = deep_merge(merged, raw_config)
    logger.info(f"Loaded merged config from {path}")
    return merged

def resolve_config_path(config_name: str) -> Path:
    p = Path(config_name)

    if p.suffix and p.suffix not in (".yaml", ".yml"):
        raise_and_log(f"Config name '{config_name}' has unexpected extension '{p.suffix}', expected .yaml/.yml")
    if not p.suffix:
        p = p.with_suffix(".yaml")
    if p.is_absolute() or p.exists():
        logger.info(f"Config path {p} is absolute or exists")
        return p

    real_path = CONFIG_ROOT / p
    if not real_path.exists():
        raise_and_log(f"Config path {real_path} does not exist", FileNotFoundError)

    logger.info(f"Resolving config path {p} to {real_path}")
    return real_path


def resolve_output_path(output_dir: str, run_name: str) -> Path:
    if (run_name := run_name.strip()) == "":
        raise_and_log("Run name cannot be empty")

    output_path = CONFIG_ROOT / output_dir / run_name
    logger.info(f"Set output path to be {output_path}")
    return output_path


def raise_and_log(msg: str, exc_type: type[Exception] = ValueError) -> NoReturn:
    logger.error(msg)
    raise exc_type(msg)


def initializer(func: Callable) -> Callable:
    """
    Automatically assigns constructor parameters as attributes.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    spec = inspect.getfullargspec(func)
    names = spec.args
    defaults = spec.defaults or ()

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for name, arg in list(zip(names[1:], args)) + list(kwargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kwargs)

    return wrapper


T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._registry:
                raise_and_log(f"{self.kind} name '{name}' is already registered to {self._registry[name].__name__}")
            self._registry[name] = cls
            logger.debug(f"Registered {self.kind} '{name}' -> {cls.__name__}")
            return cls

        return decorator

    def get(self, name: str) -> type[T]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry)) or "(none registered)"
            raise_and_log(f"{self.kind} '{name}' is not registered. Available: {available}")
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> T:
        cls = self.get(name)
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise_and_log(f"{self.kind} '{name}' has invalid configuration: {kwargs} ({e})")

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def names(self) -> list[str]:
        return sorted(self._registry)

    def __getitem__(self, key: str):
        if key not in self._registry:
            raise KeyError(f"'{key}' is not registered.")
        return self._registry[key]