import csv
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from engines.interfaces.irunner import CallBack, IRunner
from engines.registries import CALLBACK_REGISTRY
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class BaseCallBack(ABC):
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        ...
    
    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        ...
    
    def on_training_end(self, runner: IRunner) -> None:
        ...

@CALLBACK_REGISTRY.register("checkpoint")
class CheckpointCallBack(BaseCallBack):
    LATEST_CHECKPOINT_NAME = "latest.pt"
    FINAL_CHECKPOINT_NAME = "final.pt"
    
    def __init__(self, output_path: str | Path, save_every_steps: int = 1000, keep_every_epochs: int = 10) -> None:
        super().__init__(output_path)
        if save_every_steps <= 0:
            raise ValueError(f"save_every_steps must be > 0, got {save_every_steps}")
        if keep_every_epochs <= 0:
            raise ValueError(f"keep_every_epochs must be > 0, got {keep_every_epochs}")
        self.save_every_steps = save_every_steps
        self.keep_every_epochs = keep_every_epochs
    
    def _save(self, runner: IRunner, file_name: str) -> None:
        path = self.output_path / file_name
        tmp_path = path.with_suffix(f"{path.suffix}.tmp")
        try:
            torch.save(runner.state_dict(), tmp_path)
            tmp_path.replace(path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            logger.error(f"Failed to save checkpoint '{path}': {e}")
            
    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        if step > 0 and step % self.save_every_steps == 0:
            self._save(runner, self.LATEST_CHECKPOINT_NAME)
            
    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        self._save(runner, self.LATEST_CHECKPOINT_NAME)
        
        if epoch > 0 and epoch % self.keep_every_epochs == 0:
            self._save(runner, f"epoch_{epoch}.pt")
    
    def on_training_end(self, runner: IRunner) -> None:
        self._save(runner, self.FINAL_CHECKPOINT_NAME)

@CALLBACK_REGISTRY.register("logging")
class LoggingCallBack(BaseCallBack):
    def __init__(self, output_path: str | Path, log_every_steps: int = 50) -> None:
        super().__init__(output_path)
        if log_every_steps <= 0:
            raise ValueError(f"log_every_steps must be > 0, got {log_every_steps}")
        self.log_every_steps = log_every_steps
        self._epoch_metric_sums: dict[str, float] = {}
        self._epoch_step_count: int = 0

    def _to_float(self, val: Any) -> float:
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)
        
    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            try:
                val_float = self._to_float(v)
                self._epoch_metric_sums[k] = self._epoch_metric_sums.get(k, 0.0) + val_float
            except (TypeError, ValueError):
                pass
        self._epoch_step_count += 1

        if step % self.log_every_steps == 0:
            formatted_metrics = []
            for k, v in metrics.items():
                try:
                    vf = self._to_float(v)
                    formatted_metrics.append(f"{k}={vf:.4f}")
                except (TypeError, ValueError):
                    formatted_metrics.append(f"{k}={v}")
            metrics_str = ", ".join(formatted_metrics)
            logger.info(f"step={step} {metrics_str}")
        
    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        if self._epoch_step_count > 0:
            avg_metrics = {
                k: v / self._epoch_step_count for k, v in self._epoch_metric_sums.items()
            }
            avg_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items())
            logger.info(f"Epoch {epoch} complete — avg({avg_str})")

        self._epoch_metric_sums = {}
        self._epoch_step_count = 0
    
    def on_training_end(self, runner: IRunner) -> None:
        logger.info("Finished")


@CALLBACK_REGISTRY.register("early_stopping")
class EarlyStoppingCallBack(BaseCallBack):
    def __init__(
        self,
        output_path: str | Path,
        monitor: str = "loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        check_on: str = "epoch",
    ) -> None:
        super().__init__(output_path)
        if mode not in ("min", "max"):
            logger.error(f"Invalid mode '{mode}' for EarlyStoppingCallBack. Expected 'min' or 'max'.")
            raise ValueError(f"Invalid mode '{mode}' for EarlyStoppingCallBack. Expected 'min' or 'max'.")
        if check_on not in ("epoch", "step"):
            logger.error(f"Invalid check_on '{check_on}' for EarlyStoppingCallBack. Expected 'epoch' or 'step'.")
            raise ValueError(f"Invalid check_on '{check_on}' for EarlyStoppingCallBack. Expected 'epoch' or 'step'.")
        if patience <= 0:
            raise ValueError(f"patience must be > 0, got {patience}")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.check_on = check_on
        self.wait_count: int = 0
        self.best_score: float | None = None

        self._epoch_metric_sum: float = 0.0
        self._epoch_step_count: int = 0

    def _to_float(self, val: Any) -> float:
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def _evaluate(self, runner: IRunner, value: float) -> None:
        if math.isnan(value) or math.isinf(value):
            logger.warning(
                f"Early stopping monitor '{self.monitor}' received invalid value '{value}'."
            )
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(
                    f"Early stopping triggered: '{self.monitor}' invalid or did not improve for {self.wait_count} checks."
                )
                setattr(runner, "should_stop", True)
            return

        if self.best_score is None:
            self.best_score = value
            return

        is_improved = (
            (value < self.best_score - self.min_delta)
            if self.mode == "min"
            else (value > self.best_score + self.min_delta)
        )

        if is_improved:
            self.best_score = value
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(
                    f"Early stopping triggered: '{self.monitor}' did not improve for {self.wait_count} checks."
                )
                setattr(runner, "should_stop", True)

    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        if self.monitor in metrics:
            val = self._to_float(metrics[self.monitor])
            if self.check_on == "step":
                self._evaluate(runner, val)
            elif self.check_on == "epoch":
                self._epoch_metric_sum += val
                self._epoch_step_count += 1

    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        if self.check_on == "epoch":
            if self._epoch_step_count > 0:
                avg_val = self._epoch_metric_sum / self._epoch_step_count
                self._evaluate(runner, avg_val)
                self._epoch_metric_sum = 0.0
                self._epoch_step_count = 0
            elif hasattr(runner, "_epoch_metric_sums") and self.monitor in runner._epoch_metric_sums:
                step_cnt = getattr(runner, "_epoch_step_count", 1)
                avg_val = runner._epoch_metric_sums[self.monitor] / max(step_cnt, 1)
                self._evaluate(runner, float(avg_val))


@CALLBACK_REGISTRY.register("csv_logger")
class CSVLoggerCallBack(BaseCallBack):
    def __init__(self, output_path: str | Path, filename: str = "metrics.csv") -> None:
        super().__init__(output_path)
        self.csv_path = self.output_path / filename
        self._header_written: bool = self.csv_path.exists()
        self._fieldnames: list[str] = []

    def _to_val(self, val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            return float(val.item()) if val.numel() == 1 else str(val.tolist())
        return val

    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        if not metrics:
            return

        epoch = getattr(runner, "current_epoch", 0)
        clean_metrics = {k: self._to_val(v) for k, v in metrics.items()}
        current_keys = list(clean_metrics.keys())

        try:
            file_exists = self.csv_path.exists()
            if not file_exists or not self._header_written:
                self._fieldnames = ["epoch", "step"] + current_keys
                with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    self._header_written = True
                    row = {"epoch": epoch, "step": step, **clean_metrics}
                    writer.writerow(row)
            else:
                if not self._fieldnames:
                    with open(self.csv_path, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        self._fieldnames = next(reader, ["epoch", "step"] + current_keys)

                with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                    row = {"epoch": epoch, "step": step, **clean_metrics}
                    writer.writerow(row)
        except Exception as e:
            logger.error(f"Failed to write to CSV log '{self.csv_path}': {e}")


@CALLBACK_REGISTRY.register("lr_monitor")
class LRMonitorCallBack(BaseCallBack):
    def __init__(self, output_path: str | Path, log_every_steps: int = 50) -> None:
        super().__init__(output_path)
        if log_every_steps <= 0:
            raise ValueError(f"log_every_steps must be > 0, got {log_every_steps}")
        self.log_every_steps = log_every_steps

    def _get_learning_rates(self, runner: IRunner) -> dict[str, float]:
        lrs: dict[str, float] = {}
        opts: dict[str, Any] = {}
        try:
            if hasattr(runner, "_get_optimizers"):
                opts = runner._get_optimizers()
            elif hasattr(runner, "optimizer"):
                opts = {"optimizer": getattr(runner, "optimizer")}
        except Exception:
            pass

        if not isinstance(opts, dict):
            return lrs

        for opt_name, opt in opts.items():
            if hasattr(opt, "param_groups"):
                for idx, group in enumerate(opt.param_groups):
                    key = f"lr/{opt_name}_group_{idx}" if len(opt.param_groups) > 1 else f"lr/{opt_name}"
                    lrs[key] = float(group.get("lr", 0.0))
        return lrs

    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        if step % self.log_every_steps == 0:
            lrs = self._get_learning_rates(runner)
            if lrs:
                lr_str = ", ".join(f"{k}={v:.6e}" for k, v in lrs.items())
                logger.info(f"step={step} {lr_str}")

    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        lrs = self._get_learning_rates(runner)
        if lrs:
            lr_str = ", ".join(f"{k}={v:.6e}" for k, v in lrs.items())
            logger.info(f"Epoch {epoch} LR — {lr_str}")


@CALLBACK_REGISTRY.register("timer")
class TimerCallBack(BaseCallBack):
    def __init__(self, output_path: str | Path, log_every_steps: int = 50) -> None:
        super().__init__(output_path)
        if log_every_steps <= 0:
            raise ValueError(f"log_every_steps must be > 0, got {log_every_steps}")
        self.log_every_steps = log_every_steps
        self._start_time: float = time.perf_counter()
        self._step_start_time: float = time.perf_counter()
        self._epoch_start_time: float = time.perf_counter()

    def on_step_end(self, runner: IRunner, metrics: dict[str, float], step: int) -> None:
        now = time.perf_counter()
        step_duration = now - self._step_start_time
        self._step_start_time = now

        if step > 0 and step % self.log_every_steps == 0:
            logger.info(f"step={step} step_time={step_duration:.4f}s")

    def on_epoch_end(self, runner: IRunner, epoch: int) -> None:
        now = time.perf_counter()
        epoch_duration = now - self._epoch_start_time
        total_duration = now - self._start_time
        self._epoch_start_time = now

        logger.info(f"Epoch {epoch} time={epoch_duration:.2f}s total_elapsed={total_duration:.2f}s")


def build_callbacks(callback_configs: dict[str, dict[str, Any] | bool | None] | None, output_path: str | Path) -> list[CallBack]:
    callbacks: list[CallBack] = []
    if callback_configs is None:
        logger.warning("No callbacks were built. Please check your configuration if this is not intended.")
        return callbacks
    
    output_path = Path(output_path)
    for name, config in callback_configs.items():
        if config is None:
            params: dict[str, Any] = {}
        elif isinstance(config, bool):
            if not config:
                logger.warning(f"Callback '{name}' is disabled")
                continue
            params = {}
        else:
            params = dict(config)
        
        params.setdefault("output_path", output_path)
        callbacks.append(CALLBACK_REGISTRY.build(name, **params))
        
    if not callbacks:
        logger.warning("No callbacks were built. Please check your configuration if this is not intended.")
    else:
        logger.info(f"Built callbacks: {', '.join([type(cb).__name__ for cb in callbacks])}")
        
    return callbacks

@CALLBACK_REGISTRY.register("plot")
class PlotCallBack(BaseCallBack):
    def __init__(
        self,
        output_path: str | Path,
        plot_every_epochs: int = 5,
        metrics_to_plot: list[str] | None = None,
        dpi: int = 120,
    ) -> None:
        super().__init__(output_path)
        self.plot_every_epochs = plot_every_epochs
        self.metrics_to_plot = metrics_to_plot
        self.dpi = dpi

        self._history: dict[str, list[float]] = defaultdict(list)
        self._steps: list[int] = []
        self._step_metrics: dict[str, list[float]] = defaultdict(list)

        self._test_step_metrics: dict[str, list[float]] = defaultdict(list)
        self._y_true: list[np.ndarray] = []
        self._y_pred: list[np.ndarray] = []

    def on_step_end(self, runner, metrics, step):
        for k, v in metrics.items():
            try:
                self._step_metrics[k].append(float(v))
            except (TypeError, ValueError):
                pass
        if metrics:
            self._steps.append(step)

        last = getattr(runner, "last_outputs", None)
        if last is not None:
            if last.get("y_pred") is not None:
                self._y_pred.append(np.asarray(last["y_pred"]))
            if last.get("y_true") is not None:
                self._y_true.append(np.asarray(last["y_true"]))

    def on_epoch_end(self, runner, epoch):
        for k, vals in self._step_metrics.items():
            if vals:
                self._history[k].append(float(np.mean(vals)))
        self._step_metrics.clear()

        if epoch % self.plot_every_epochs == 0 or epoch == 0:
            self._plot_curves(epoch)

    def on_training_end(self, runner):
        self._plot_curves(runner.current_epoch, final=True)
        self._plot_scatter()

    def _plot_curves(self, epoch: int, final: bool = False):
        if not self._history:
            return
        metrics = self.metrics_to_plot or list(self._history.keys())
        fig, ax = plt.subplots(figsize=(7, 4))
        for k in metrics:
            if k in self._history and self._history[k]:
                ax.plot(range(1, len(self._history[k]) + 1),
                        self._history[k], label=k, marker="o", markersize=3)
        ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.legend()
        ax.set_title("Training curves")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        suffix = "final" if final else f"epoch{epoch}"
        fig.savefig(self.output_path / f"train_curves_{suffix}.png", dpi=self.dpi)
        plt.close(fig)

    def _plot_scatter(self):
        if not self._y_true or not self._y_pred:
            return
        yt = np.concatenate(self._y_true, axis=0).reshape(-1)
        yp = np.concatenate(self._y_pred, axis=0).reshape(-1)
        if yt.shape != yp.shape:
            logger.warning(f"shape mismatch y_true={yt.shape} y_pred={yp.shape}, skip scatter")
            return
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yt, yp, s=4, alpha=0.5)
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, "r--", linewidth=1)
        ax.set_xlabel("y_true"); ax.set_ylabel("y_pred")
        ax.set_title("y_true vs y_pred"); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.output_path / "test_scatter.png", dpi=self.dpi)
        plt.close(fig)
        logger.info(f"Saved scatter to {self.output_path / 'test_scatter.png'}")