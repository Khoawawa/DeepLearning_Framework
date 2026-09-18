from abc import ABC, abstractmethod
from pathlib import Path

from typing import Any
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from engines.interfaces.icommon import Encoder, Predictor, MaskSampler, Criterion, TorchModule
from tqdm import tqdm
from engines.spec import TesterBuildSpec
from loguru import logger
from engines.interfaces.icommon import TorchModule
from engines.interfaces.irunner import CallBack
from engines.registries import TESTER_BUILDER_REGISTRY, MODEL_REGISTRY
from engines.engine_utils import to_var


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
                        cb.on_step_end(metrics, step)

        except Exception:
            logger.exception("Evaluation failed during testing run")
            raise
        finally:
            for cb in call_backs:
                cb.on_test_end(self)

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
    def build_unique_kwargs(cls, cfg: dict[str, Any]) -> TesterBuildSpec:
        ...

    @abstractmethod
    def _to_components(self, device: torch.device) -> None:
        ...

    def to(self, device: torch.device) -> "BaseTester":
        self.device = device
        self._to_components(device)
        return self

@TESTER_BUILDER_REGISTRY.register("simple")
class SimpleTester(BaseTester):
    def __init__(self, model: nn.Module, criterion: nn.Module | None = None):
        super().__init__()
        self.model = model
        self.criterion = criterion or nn.MSELoss()

    @classmethod
    def required_components(cls) -> list[str]:
        return ["model"]

    @classmethod
    def build_unique_kwargs(cls, cfg: dict[str, Any]) -> TesterBuildSpec:
        model_cfg = cfg["model"]
        model_name = model_cfg["name"].lower()
        model = MODEL_REGISTRY.build(model_name, **model_cfg.get("params", {}))

        return {
            "model": model,
        }
    def test_step(self, batch: Any) -> dict[str, float]:
        x, y = batch
        preds = self.model(x)
        loss = self.criterion(preds, y)
        return {"loss": loss.item()}

    def _eval_mode(self) -> None:
        self.model.eval()

    def _get_component_state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict()}

    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state)

    def _to_components(self, device: torch.device) -> None:
        self.model.to(device)

if __name__ == "__main__":
    model = nn.Linear(4, 1)
    tester = SimpleTester(model=model)
    
    x = torch.randn(10, 4)
    y = torch.randn(10, 1)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    
    metrics = tester.test(loader)

    print(metrics)

    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert not model.training 
