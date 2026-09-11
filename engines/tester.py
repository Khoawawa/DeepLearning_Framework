from abc import ABC, abstractmethod
from pathlib import Path

from typing import Any
import torch
from torch.utils.data import DataLoader
from engines.interfaces.icommon import Encoder, Predictor, MaskSampler, Criterion, TorchModule
from tqdm import tqdm
from engines.spec import TesterBuildSpec
from loguru import logger
from engines.interfaces.icommon import CallBack, TorchModule

class BaseTester(ABC):
    def __init__(self) -> None:
        self.device: torch.device = torch.device("cpu")
        self.allowed_checkpoint_format: list[str] = [".pth", ".pkl", ".pt"]

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
                    batch = self._prepare_batch(batch)
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
                cb.on_training_end()

        if total_samples == 0:
            logger.warning("No samples were evaluated")
            return {}

        return {k: v / total_samples for k, v in aggregated_metrics.items()}

    def _prepare_batch(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, dict):
            return {
                k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                v.to(self.device) if isinstance(v, torch.Tensor) else v
                for v in batch
            )
        return batch

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
    @classmethod
    def required_components(cls) -> list[str]:
        ...

    @abstractmethod
    @classmethod
    def build_unique_kwargs(cls, cfg: dict[str, Any]) -> TesterBuildSpec:
        ...

    @abstractmethod
    def _to_components(self, device: torch.device) -> None:
        ...

    def to(self, device: torch.device) -> "BaseTester":
        self.device = device
        self._to_components(device)
        return self

class IJepaTester(BaseTester):
    def __init__(
        self,
        context_encoder: Encoder,
        target_encoder: Encoder,
        predictor: Predictor,
        mask_sampler: MaskSampler,
        criterion: Criterion,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.mask_sampler = mask_sampler
        self.criterion = criterion

    def _eval_mode(self) -> None:
        self.context_encoder.eval()
        self.target_encoder.eval()
        self.predictor.eval()
        self.mask_sampler.eval()
        self.criterion.eval()

    def test_step(self, x: Any) -> dict[str, float]:
        if isinstance(x, dict):
            x = x.get("image", x.get("x"))

        ctx_idx, tgt_idx_list = self.mask_sampler.sample(x)
        context_reprs = self.context_encoder(x, ctx_idx)

        full_target_reprs = self.target_encoder(x)
        target_reprs_list = [
            self._gather(full_target_reprs, idx) for idx in tgt_idx_list
        ]
        preds = [self.predictor(context_reprs, idx) for idx in tgt_idx_list]

        losses = [
            self.criterion(p, t) for p, t in zip(preds, target_reprs_list)
        ]
        loss = torch.stack(losses).sum()

        return {"val_loss": loss.item()}

    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        self.context_encoder.load_state_dict(state["context_encoder"])
        self.target_encoder.load_state_dict(state["target_encoder"])
        self.predictor.load_state_dict(state["predictor"])
        self.mask_sampler.load_state_dict(state["mask_sampler"])
        self.criterion.load_state_dict(state["criterion"])

    def _to_components(self, device: torch.device) -> None:
        self.context_encoder.to(device)
        self.target_encoder.to(device)
        self.predictor.to(device)
        self.mask_sampler.to(device)
        self.criterion.to(device)

TESTER_BUILDER_REGISTRY: dict[str, type[BaseTester]] = {
    "ijepa": IJepaTester,
}