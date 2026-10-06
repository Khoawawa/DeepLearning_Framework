from typing import Any, Iterable, Mapping, Sequence
from loguru import logger


class Validation:

    @staticmethod
    def require_keys(
        cfg: Mapping[str, Any],
        required: Iterable[str],
        *,
        context: str = "config",
    ) -> None:
        missing = [k for k in required if k not in cfg]
        if missing:
            msg = f"{context} missing required keys: {missing}"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def require_type(
        value: Any,
        expected: type | tuple[type, ...],
        *,
        name: str,
    ) -> None:
        if not isinstance(value, expected):
            msg = f"'{name}' must be {expected}, got {type(value).__name__}"
            logger.error(msg)
            raise TypeError(msg)

    @staticmethod
    def require_positive(value: int | float, *, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"'{name}' must be int|float, got {type(value).__name__}"
            logger.error(msg)
            raise TypeError(msg)
        if value <= 0:
            msg = f"'{name}' must be > 0, got {value}"
            logger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def require_non_empty(value: Sequence[Any] | str, *, name: str) -> None:
        if len(value) == 0:
            msg = f"'{name}' must not be empty"
            logger.error(msg)
            raise ValueError(msg)