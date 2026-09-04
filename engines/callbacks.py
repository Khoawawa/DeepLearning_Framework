from pathlib import Path

import torch

from engines.interfaces.icommon import CallBack
from loguru import logger
import os

from engines.interfaces.irunner import ITrainer

class CheckpointCallBack:
    LATEST_CHECKPOINT_NAME = "latest.pt"
    FINAL_CHECKPOINT_NAME = "final.pt"
    
    def __init__(self, trainer: ITrainer, output_dir: str, save_every_steps: int = 1000, keep_every_epochs: int = 10) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.trainer = trainer
        self.output_dir = Path(output_dir)
        self.save_every_steps = save_every_steps
        self.keep_every_epochs = keep_every_epochs
    
    def _save(self, file_name: str) -> None:
        path = self.output_dir / file_name
        try:
            torch.save(self.trainer.state_dict(), path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint '{path}': {e}")
            
    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if step > 0 and step % self.save_every_steps == 0:
            self._save(self.LATEST_CHECKPOINT_NAME)
            
    
    def on_epoch_end(self, epoch: int) -> None:
        self._save(self.LATEST_CHECKPOINT_NAME)
        
        if epoch > 0 and epoch % self.keep_every_epochs == 0:
            self._save(f"epoch_{epoch}.pt")
    
    def on_training_end(self) -> None:
        self._save(self.FINAL_CHECKPOINT_NAME)

class LoggingCallBack:
    def __init__(self, log_every_steps: int = 50) -> None:
        self.log_every_steps = log_every_steps
        self._epoch_metric_sums: dict[str, float] = {}
        self._epoch_step_count: int = 0
        
    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self._epoch_metric_sums[k] = self._epoch_metric_sums.get(k, 0.0) + v
        self._epoch_step_count += 1

        if step % self.log_every_steps == 0:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            logger.info(f"step={step} {metrics_str}")
        
    def on_epoch_end(self, epoch: int) -> None:
        if self._epoch_step_count > 0:
            avg_metrics = {
                k: v / self._epoch_step_count for k, v in self._epoch_metric_sums.items()
            }
            avg_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items())
            logger.info(f"Epoch {epoch} complete — avg({avg_str})")

        self._epoch_metric_sums = {}
        self._epoch_step_count = 0
    
    def on_training_end(self) -> None:
        logger.info("Finished")
        