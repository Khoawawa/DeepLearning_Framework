from abc import ABC, abstractmethod
from pathlib import Path

from typing import Any, Callable
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from engines.interfaces.icommon import Encoder, Predictor, MaskSampler, Criterion, TorchModule
from tqdm import tqdm
from engines.spec import TesterBuildSpec
from loguru import logger
from engines.interfaces.icommon import TorchModule
from engines.interfaces.irunner import CallBack
from engines.registries import TESTER_BUILDER_REGISTRY, MODEL_REGISTRY, CRITERION_REGISTRY
from engines.engine_utils import to_var
from utils import raise_and_log

from utils.common_utils import load_config, load_yaml



class BaseTester(ABC):
    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.allowed_checkpoint_format: list[str] = [".pth", ".pkl", ".pt"]

    def state_dict(self) -> dict[str, Any]:
        return self._get_component_state_dict()

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        self._load_component_state_dict(state)

    def load_weights(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint {checkpoint_path} not found")
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

        logger.info(f"Loading weights from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self._load_component_state_dict(state)

    def test(self, data_loader: DataLoader, checkpoint_path: str | Path | None = None, call_backs: list[CallBack] | None = None,
    ) -> dict[str, float]:
        if checkpoint_path is not None:
            self.load_weights(checkpoint_path)

        self._eval_mode()
        call_backs = call_backs or []
        aggregated_metrics: dict[str, float] = {}
        total_samples: int = 0

        try:
            with torch.no_grad():
                for step, batch in enumerate(tqdm(data_loader, desc="Testing")):
                    batch = to_var(batch, self.device)
                    metrics = self.test_step(batch)

                    batch_size = self._infer_batch_size(batch)
                    total_samples += batch_size

                    for k, v in metrics.items():
                        aggregated_metrics[k] = aggregated_metrics.get(k, 0.0) + (v * batch_size)

                    for cb in call_backs:
                        cb.on_step_end(self, metrics, step)

        except Exception:
            logger.exception("Evaluation failed during testing run")
            raise
        finally:
            for cb in call_backs:
                cb.on_testing_end(self)

        if total_samples == 0:
            logger.warning("No samples were evaluated")
            return {}

        return {k: v / total_samples for k, v in aggregated_metrics.items()}

    def _infer_batch_size(self, batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, dict):
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
        raise ValueError("Cannot infer batch size from batch") 

    @abstractmethod
    def test_step(self, x: Any) -> dict[str, float]:
        ...

    @abstractmethod
    def _eval_mode(self) -> None:
        ...

    @abstractmethod
    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def _get_component_state_dict(self) -> dict[str, Any]:
        ...

    @classmethod
    @abstractmethod
    def required_components(cls) -> list[str]:
        ...

    @classmethod
    @abstractmethod
    def _build_unique_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        ...

    @classmethod
    def build_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        unique = cls._build_unique_kwargs(cfg)
        return {**unique}

    @abstractmethod
    def _to_components(self, device: torch.device) -> None:
        ...

    def to(self, device: torch.device) -> "BaseTester":
        self.device = device
        self._to_components(device)
        return self

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[Any, Any | None]:
        """Tách batch thành (input, label). Hỗ trợ dict / tuple / list / tensor."""
        if isinstance(batch, dict):
            inp = batch.get("image", batch.get("x", next(iter(batch.values()))))
            label = batch.get("label", batch.get("y", None))
        elif isinstance(batch, (list, tuple)):
            inp = batch[0]
            label = batch[1] if len(batch) > 1 else None
        else:
            inp, label = batch, None

        if not isinstance(inp, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        return inp, label

    def _compute_loss(
        self,
        out: torch.Tensor,
        label: torch.Tensor,
        criterion: nn.Module | Callable | None = None,
    ) -> torch.Tensor:
        """Tính loss. Trả về tensor scalar (chưa detach)."""
        criterion = criterion or getattr(self, "criterion", None)
        if criterion is None:
            raise ValueError("No criterion available")
        return criterion(out, label)

@TESTER_BUILDER_REGISTRY.register("cnn")
class CNNTester(BaseTester):
    def __init__(self, cnn_block: Encoder, criterion: Criterion | None = None):
        super().__init__()
        self.cnn_block = cnn_block
        self.criterion = criterion or nn.MSELoss()

    @classmethod
    def required_components(cls) -> list[str]:
        return ["cnn_block"]

    @classmethod
    def _build_unique_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        kwargs = {}
        try:
            kwargs["cnn_block"] = nn.Conv2d(**cfg["cnn_block"])
        except KeyError as e:
            raise_and_log(f"Missing required cnn config key: {e}")
        except TypeError as e:
            raise_and_log(f"Invalid cnn config: {e}")

        if "criteria" in cfg and "main" in cfg["criteria"]:
            try:
                crit_cfg = dict(cfg["criteria"]["main"])
                crit_cls_name = crit_cfg.pop("type", "mse")
                crit_cls = CRITERION_REGISTRY.get(crit_cls_name)
                kwargs["criterion"] = crit_cls(**crit_cfg)
            except KeyError as e:
                raise_and_log(f"Missing required criterion config key: {e}")
            except (TypeError, AttributeError) as e:
                raise_and_log(f"Invalid criterion config: {e}")

        return kwargs

    def test_step(self, batch: Any) -> dict[str, float]:
        inp, label = self._unpack_batch(batch)
        out = self.cnn_block(inp)

        if label is not None and isinstance(label, torch.Tensor):
            loss = self._compute_loss(out, label)
            return {"loss": float(loss.item())}

        return {"output_mean": float(out.mean().item())}

    def _eval_mode(self) -> None:
        self.cnn_block.eval()

    def _get_component_state_dict(self) -> dict[str, Any]:
        return {"cnn_block": self.cnn_block.state_dict()}

    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        cnn_block_state = state.get("cnn_block", state)
        self.cnn_block.load_state_dict(cnn_block_state)

    def _to_components(self, device: torch.device) -> None:
        self.cnn_block.to(device)


# if __name__ == "__main__":
#     cfg = load_yaml("configs/config.yaml")
#     cnn_tester = CNNTester(**CNNTester.build_kwargs(cfg))

#     x = torch.randn(4, 3, 32, 32)
#     y = torch.randn(4, 32, 30, 30)
#     loader = DataLoader(TensorDataset(x, y), batch_size=2)
#     print(cnn_tester.test(loader))


