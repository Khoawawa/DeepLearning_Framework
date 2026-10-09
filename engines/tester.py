from __future__ import annotations

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
from engines.registries import TESTER_BUILDER_REGISTRY, CRITERION_REGISTRY
from engines.engine_utils import to_var
from utils import raise_and_log

from utils.common_utils import load_config, load_yaml



class BaseTester(ABC):
    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.should_stop: bool = False
        self.device: torch.device = torch.device("cpu")
        self.allowed_checkpoint_format: list[str] = [".pth", ".pkl", ".pt"]
        self._epoch_metric_sums: dict[str, float] = {}
        self._epoch_step_count: int = 0
        self.last_outputs: dict[str, Any] | None = None

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
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self._load_component_state_dict(state)

    def test(self, data_loader: DataLoader, checkpoint_path: str | Path | None = None, call_backs: list[CallBack] | None = None,
    ) -> dict[str, float]:
        if checkpoint_path is not None:
            self.load_weights(checkpoint_path)

        self._eval_mode()
        call_backs = call_backs or []
        aggregated_metrics: dict[str, float] = {}
        total_samples: int = 0

        self.should_stop = False
        self._epoch_metric_sums = {}
        self._epoch_step_count = 0

        try:
            with torch.no_grad():
                for step, batch in enumerate(tqdm(data_loader, desc="Testing")):
                    batch = to_var(batch, self.device)
                    metrics = self.test_step(batch)
                    batch_size = self._infer_batch_size(batch)
                    total_samples += batch_size

                    # tích lũy giống LoggingCallBack/EarlyStopping làm với trainer
                    for k, v in metrics.items():
                        try:
                            vf = float(v.item() if isinstance(v, torch.Tensor) else v)
                            self._epoch_metric_sums[k] = self._epoch_metric_sums.get(k, 0.0) + vf
                        except (TypeError, ValueError):
                            pass
                    self._epoch_step_count += 1

                    for k, v in metrics.items():
                        aggregated_metrics[k] = aggregated_metrics.get(k, 0.0) + (v * batch_size)

                    for cb in call_backs:
                        cb.on_step_end(self, metrics, self.global_step)

                    self.global_step += 1
                    if self.should_stop:
                        logger.info(f"Early stopping requested at step {self.global_step}")
                        break

            self.current_epoch += 1

            for cb in call_backs:
                cb.on_epoch_end(self, self.current_epoch - 1)

        except Exception:
            logger.exception("Evaluation failed during testing run")
            raise
        finally:
            for cb in call_backs:
                cb.on_training_end(self)

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

        self.last_outputs = {
            "y_pred": out.detach().cpu().numpy(),
            "y_true": label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else None,
        }

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


class ModalityTester(BaseTester):
    """Base tester for single and multi-modality model evaluation."""

    def __init__(self, model: nn.Module, criterion: Criterion | None = None):
        super().__init__()
        self.model = model
        self.criterion = criterion or nn.CrossEntropyLoss()

    @classmethod
    def required_components(cls) -> list[str]:
        return []

    @classmethod
    def _build_unique_kwargs(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        kwargs = {}
        if "model_instance" in cfg:
            kwargs["model"] = cfg["model_instance"]
        elif "model_class" in cfg:
            model_cls = cfg["model_class"]
            model_kwargs = cfg.get("model_params", {})
            kwargs["model"] = model_cls(**model_kwargs)
        elif "model" in cfg and isinstance(cfg["model"], dict):
            from models.transformer.model import TransformerModel
            kwargs["model"] = TransformerModel(**cfg["model"])
        elif "model" in cfg and isinstance(cfg["model"], nn.Module):
            kwargs["model"] = cfg["model"]
        else:
            raise_and_log("Config must specify 'model_instance', 'model_class', or 'model'")

        if "criteria" in cfg and "main" in cfg["criteria"]:
            try:
                crit_cfg = dict(cfg["criteria"]["main"])
                crit_cls_name = crit_cfg.pop("type", "cross_entropy")
                crit_cls = CRITERION_REGISTRY.get(crit_cls_name)
                kwargs["criterion"] = crit_cls(**crit_cfg)
            except Exception as e:
                raise_and_log(f"Invalid criterion config: {e}")

        return kwargs

    def _eval_mode(self) -> None:
        self.model.eval()

    def _get_component_state_dict(self) -> dict[str, Any]:
        return {"model": self.model.state_dict()}

    def _load_component_state_dict(self, state: dict[str, Any]) -> None:
        model_state = state.get("model", state)
        self.model.load_state_dict(model_state)

    def _to_components(self, device: torch.device) -> None:
        self.model.to(device)


@TESTER_BUILDER_REGISTRY.register("vit")
class ViTTester(ModalityTester):
    """Tester for Vision Transformer models."""

    def test_step(self, batch: Any) -> dict[str, float]:
        inp, label = self._unpack_batch(batch)
        out = self.model(inp)
        self.last_outputs = {
            "y_pred": out.detach().cpu().numpy(),
            "y_true": label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else None,
        }
        if label is not None and isinstance(label, torch.Tensor):
            loss = self._compute_loss(out, label)
            return {"loss": float(loss.item())}
        return {"output_mean": float(out.mean().item())}


@TESTER_BUILDER_REGISTRY.register("text")
class TextTester(ModalityTester):
    """Tester for Text Transformer models."""

    def test_step(self, batch: Any) -> dict[str, float]:
        if isinstance(batch, dict):
            inp = batch.get("text", batch.get("x", next(iter(batch.values()))))
            label = batch.get("label", batch.get("y", None))
        else:
            inp, label = self._unpack_batch(batch)

        out = self.model(inp)
        self.last_outputs = {
            "y_pred": out.detach().cpu().numpy(),
            "y_true": label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else None,
        }
        if label is not None and isinstance(label, torch.Tensor):
            loss = self._compute_loss(out, label)
            return {"loss": float(loss.item())}
        return {"output_mean": float(out.mean().item())}


@TESTER_BUILDER_REGISTRY.register("audio")
class AudioTester(ModalityTester):
    """Tester for Audio Transformer models."""

    def test_step(self, batch: Any) -> dict[str, float]:
        if isinstance(batch, dict):
            inp = batch.get("audio", batch.get("x", next(iter(batch.values()))))
            label = batch.get("label", batch.get("y", None))
        else:
            inp, label = self._unpack_batch(batch)

        out = self.model(inp)
        self.last_outputs = {
            "y_pred": out.detach().cpu().numpy(),
            "y_true": label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else None,
        }
        if label is not None and isinstance(label, torch.Tensor):
            loss = self._compute_loss(out, label)
            return {"loss": float(loss.item())}
        return {"output_mean": float(out.mean().item())}


@TESTER_BUILDER_REGISTRY.register("multimodal")
class MultimodalTester(ModalityTester):
    """Tester for Multimodal models taking dict batches."""

    def test_step(self, batch: Any) -> dict[str, float]:
        label = batch.get("label", batch.get("y", None)) if isinstance(batch, dict) else None
        out = self.model(batch)

        self.last_outputs = {
            "y_pred": out.detach().cpu().numpy(),
            "y_true": label.detach().cpu().numpy() if isinstance(label, torch.Tensor) else None,
        }

        if label is not None and isinstance(label, torch.Tensor):
            loss = self._compute_loss(out, label)
            return {"loss": float(loss.item())}

        return {"output_mean": float(out.mean().item())}


@TESTER_BUILDER_REGISTRY.register("standard")
class StandardTester(ViTTester):
    """General single-modality tester fallback."""
    pass



