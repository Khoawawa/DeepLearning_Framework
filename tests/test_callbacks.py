import csv
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from engines.callbacks import (
    CSVLoggerCallBack,
    CheckpointCallBack,
    EarlyStoppingCallBack,
    LRMonitorCallBack,
    LoggingCallBack,
    TrainPlotCallBack,
    TestPlotCallBack,
    TimerCallBack,
    build_callbacks,
)
from engines.interfaces.irunner import IRunner
from engines.registries import CALLBACK_REGISTRY


def test_callback_registration() -> None:
    expected_callbacks = [
        "checkpoint",
        "logging",
        "early_stopping",
        "csv_logger",
        "lr_monitor",
        "timer",
        "train_plot",
        "test_plot",
    ]
    for name in expected_callbacks:
        assert name in CALLBACK_REGISTRY
        assert CALLBACK_REGISTRY.get(name) is not None


def test_early_stopping_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid mode"):
        EarlyStoppingCallBack(output_path=tmp_path, mode="invalid_mode")

    with pytest.raises(ValueError, match="Invalid check_on"):
        EarlyStoppingCallBack(output_path=tmp_path, check_on="invalid_check_on")

    with pytest.raises(ValueError, match="patience must be > 0"):
        EarlyStoppingCallBack(output_path=tmp_path, patience=0)


def test_early_stopping_trigger_step(tmp_path: Path) -> None:
    cb = EarlyStoppingCallBack(
        output_path=tmp_path,
        monitor="loss",
        patience=2,
        mode="min",
        check_on="step",
    )
    runner = MagicMock(spec=IRunner)
    runner.should_stop = False

    cb.on_step_end(runner, {"loss": 1.0}, step=1)
    assert not getattr(runner, "should_stop", False)

    cb.on_step_end(runner, {"loss": 1.1}, step=2)  # wait 1
    assert not getattr(runner, "should_stop", False)

    cb.on_step_end(runner, {"loss": 1.2}, step=3)  # wait 2 -> trigger stop
    assert getattr(runner, "should_stop", False)


def test_early_stopping_trigger_epoch(tmp_path: Path) -> None:
    cb = EarlyStoppingCallBack(
        output_path=tmp_path,
        monitor="loss",
        patience=2,
        mode="min",
        check_on="epoch",
    )
    runner = MagicMock(spec=IRunner)
    runner.should_stop = False

    # Epoch 0: steps with loss 1.0 and 1.2 -> avg 1.1 (best = 1.1)
    cb.on_step_end(runner, {"loss": 1.0}, step=0)
    cb.on_step_end(runner, {"loss": 1.2}, step=1)
    cb.on_epoch_end(runner, epoch=0)
    assert cb.best_score == pytest.approx(1.1)
    assert not getattr(runner, "should_stop", False)

    # Epoch 1: steps with loss 1.5 and 1.5 -> avg 1.5 (no improve, wait = 1)
    cb.on_step_end(runner, {"loss": 1.5}, step=2)
    cb.on_step_end(runner, {"loss": 1.5}, step=3)
    cb.on_epoch_end(runner, epoch=1)
    assert not getattr(runner, "should_stop", False)

    # Epoch 2: steps with loss 1.6 and 1.6 -> avg 1.6 (no improve, wait = 2 -> trigger)
    cb.on_step_end(runner, {"loss": 1.6}, step=4)
    cb.on_step_end(runner, {"loss": 1.6}, step=5)
    cb.on_epoch_end(runner, epoch=2)
    assert getattr(runner, "should_stop", False)


def test_early_stopping_max_mode(tmp_path: Path) -> None:
    cb = EarlyStoppingCallBack(
        output_path=tmp_path,
        monitor="acc",
        patience=2,
        mode="max",
        check_on="step",
    )
    runner = MagicMock(spec=IRunner)
    runner.should_stop = False

    cb.on_step_end(runner, {"acc": 0.8}, step=1)
    assert cb.best_score == 0.8

    cb.on_step_end(runner, {"acc": 0.7}, step=2)  # wait 1
    assert not getattr(runner, "should_stop", False)

    cb.on_step_end(runner, {"acc": 0.6}, step=3)  # wait 2 -> stop
    assert getattr(runner, "should_stop", False)


def test_early_stopping_nan_loss(tmp_path: Path) -> None:
    cb = EarlyStoppingCallBack(
        output_path=tmp_path,
        monitor="loss",
        patience=2,
        mode="min",
        check_on="step",
    )
    runner = MagicMock(spec=IRunner)
    runner.should_stop = False

    cb.on_step_end(runner, {"loss": 1.0}, step=1)
    assert cb.best_score == 1.0

    cb.on_step_end(runner, {"loss": float("nan")}, step=2)  # wait 1
    assert not getattr(runner, "should_stop", False)

    cb.on_step_end(runner, {"loss": float("nan")}, step=3)  # wait 2 -> trigger stop
    assert getattr(runner, "should_stop", False)


def test_csv_logger_callback(tmp_path: Path) -> None:
    cb = CSVLoggerCallBack(output_path=tmp_path, filename="test_metrics.csv")
    runner = MagicMock(spec=IRunner)
    runner.current_epoch = 0

    cb.on_step_end(runner, {"loss": 0.5, "acc": 0.9}, step=1)
    cb.on_step_end(runner, {"loss": 0.4, "acc": 0.95}, step=2)

    csv_file = tmp_path / "test_metrics.csv"
    assert csv_file.exists()

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 2
        assert reader[0]["loss"] == "0.5"
        assert reader[1]["acc"] == "0.95"


def test_csv_logger_preexisting_file(tmp_path: Path) -> None:
    csv_file = tmp_path / "metrics.csv"
    # Create pre-existing CSV
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        f.write("epoch,step,loss\n0,1,0.5\n")

    cb = CSVLoggerCallBack(output_path=tmp_path, filename="metrics.csv")
    runner = MagicMock(spec=IRunner)
    runner.current_epoch = 0

    cb.on_step_end(runner, {"loss": 0.4}, step=2)

    with open(csv_file, mode="r", encoding="utf-8") as f:
        lines = f.readlines()
        # Verify header is not duplicated
        header_count = sum(1 for line in lines if line.startswith("epoch,"))
        assert header_count == 1
        assert len(lines) == 3  # 1 header + 2 data rows


def test_csv_logger_pytorch_tensors(tmp_path: Path) -> None:
    cb = CSVLoggerCallBack(output_path=tmp_path, filename="tensor_metrics.csv")
    runner = MagicMock(spec=IRunner)
    runner.current_epoch = 0

    tensor_loss = torch.tensor(0.42, requires_grad=True)
    cb.on_step_end(runner, {"loss": tensor_loss}, step=1)

    csv_file = tmp_path / "tensor_metrics.csv"
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert float(reader[0]["loss"]) == pytest.approx(0.42)


def test_csv_logger_empty_metrics(tmp_path: Path) -> None:
    cb = CSVLoggerCallBack(output_path=tmp_path, filename="empty_test.csv")
    runner = MagicMock(spec=IRunner)

    # Calling with empty dict should return cleanly without writing header
    cb.on_step_end(runner, {}, step=1)
    assert not (tmp_path / "empty_test.csv").exists()


def test_logging_callback_with_tensors(tmp_path: Path) -> None:
    cb = LoggingCallBack(output_path=tmp_path, log_every_steps=1)
    runner = MagicMock(spec=IRunner)

    tensor_loss = torch.tensor(0.5, requires_grad=True)
    cb.on_step_end(runner, {"loss": tensor_loss}, step=1)
    assert isinstance(cb._epoch_metric_sums["loss"], float)
    assert cb._epoch_metric_sums["loss"] == 0.5
    cb.on_epoch_end(runner, epoch=0)


def test_invalid_interval_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="save_every_steps must be > 0"):
        CheckpointCallBack(output_path=tmp_path, save_every_steps=0)

    with pytest.raises(ValueError, match="keep_every_epochs must be > 0"):
        CheckpointCallBack(output_path=tmp_path, keep_every_epochs=-1)

    with pytest.raises(ValueError, match="log_every_steps must be > 0"):
        LoggingCallBack(output_path=tmp_path, log_every_steps=0)

    with pytest.raises(ValueError, match="log_every_steps must be > 0"):
        LRMonitorCallBack(output_path=tmp_path, log_every_steps=0)

    with pytest.raises(ValueError, match="log_every_steps must be > 0"):
        TimerCallBack(output_path=tmp_path, log_every_steps=0)


def test_checkpoint_callback_atomic_save(tmp_path: Path) -> None:
    cb = CheckpointCallBack(output_path=tmp_path, save_every_steps=1, keep_every_epochs=1)
    runner = MagicMock(spec=IRunner)
    runner.state_dict.return_value = {"weight": torch.ones(2, 2)}

    cb.on_step_end(runner, {}, step=1)
    assert (tmp_path / "latest.pt").exists()
    assert not (tmp_path / "latest.pt.tmp").exists()

    cb.on_epoch_end(runner, epoch=1)
    assert (tmp_path / "epoch_1.pt").exists()

    cb.on_training_end(runner)
    assert (tmp_path / "final.pt").exists()


def test_checkpoint_callback_save_error_recovery(tmp_path: Path) -> None:
    cb = CheckpointCallBack(output_path=tmp_path, save_every_steps=1)
    runner = MagicMock(spec=IRunner)

    # Make runner.state_dict return an unpicklable/invalid object to trigger save exception
    runner.state_dict.side_effect = TypeError("Unserializable object")

    # Should log error and not crash process
    cb._save(runner, "invalid.pt")
    assert not (tmp_path / "invalid.pt").exists()
    assert not (tmp_path / "invalid.pt.tmp").exists()


def test_lr_monitor_callback(tmp_path: Path) -> None:
    cb = LRMonitorCallBack(output_path=tmp_path, log_every_steps=1)
    runner = MagicMock()

    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(2, 2))], lr=0.01)
    runner._get_optimizers.return_value = {"optimizer": opt}

    lrs = cb._get_learning_rates(runner)
    assert "lr/optimizer" in lrs
    assert pytest.approx(lrs["lr/optimizer"]) == 0.01

    cb.on_step_end(runner, {}, step=1)
    cb.on_epoch_end(runner, epoch=0)


def test_lr_monitor_multiple_param_groups(tmp_path: Path) -> None:
    cb = LRMonitorCallBack(output_path=tmp_path, log_every_steps=1)
    runner = MagicMock()

    param1 = torch.nn.Parameter(torch.randn(2, 2))
    param2 = torch.nn.Parameter(torch.randn(2, 2))
    opt = torch.optim.SGD([
        {"params": [param1], "lr": 0.01},
        {"params": [param2], "lr": 0.001},
    ])
    runner._get_optimizers.return_value = {"opt": opt}

    lrs = cb._get_learning_rates(runner)
    assert "lr/opt_group_0" in lrs
    assert "lr/opt_group_1" in lrs
    assert pytest.approx(lrs["lr/opt_group_0"]) == 0.01
    assert pytest.approx(lrs["lr/opt_group_1"]) == 0.001


def test_timer_callback(tmp_path: Path) -> None:
    cb = TimerCallBack(output_path=tmp_path, log_every_steps=1)
    runner = MagicMock(spec=IRunner)

    cb.on_step_end(runner, {}, step=1)
    cb.on_epoch_end(runner, epoch=0)


def test_plot_callback_curves_and_scatter(tmp_path: Path) -> None:
    cb = TrainPlotCallBack(
        output_path=tmp_path,
        plot_every_epochs=1,
        metrics_to_plot=["loss", "acc"],
        dpi=100,
    )
    runner = MagicMock(spec=IRunner)
    runner.current_epoch = 2

    # Step 1
    runner.last_outputs = {
        "y_true": np.array([1.0, 2.0]),
        "y_pred": np.array([1.1, 1.9]),
    }
    cb.on_step_end(runner, {"loss": 0.5, "acc": 0.8}, step=1)

    # Step 2
    runner.last_outputs = {
        "y_true": np.array([3.0, 4.0]),
        "y_pred": np.array([2.9, 4.1]),
    }
    cb.on_step_end(runner, {"loss": 0.4, "acc": 0.85}, step=2)

    cb.on_epoch_end(runner, epoch=0)

    # Check curves plot saved
    epoch0_plot = tmp_path / "train_curves_epoch0.png"
    assert epoch0_plot.exists()

    # Finish training
    cb.on_training_end(runner)
    final_plot = tmp_path / "train_curves_final.png"
    scatter_plot = tmp_path / "test_scatter.png"
    assert final_plot.exists()
    assert scatter_plot.exists()


def test_plot_callback_shape_mismatch_warning(tmp_path: Path) -> None:
    cb = TrainPlotCallBack(output_path=tmp_path)
    runner = MagicMock(spec=IRunner)
    runner.current_epoch = 1

    runner.last_outputs = {
        "y_true": np.array([1.0, 2.0]),
        "y_pred": np.array([1.1, 1.9, 2.5]),  # length mismatch
    }
    cb.on_step_end(runner, {"loss": 0.5}, step=1)
    cb.on_training_end(runner)  # should log warning and skip scatter without error

    scatter_plot = tmp_path / "test_scatter.png"
    assert not scatter_plot.exists()


def test_build_callbacks(tmp_path: Path) -> None:
    configs = {
        "checkpoint": {"save_every_steps": 500},
        "logging": True,
        "early_stopping": {"patience": 3},
        "csv_logger": True,
        "timer": None,  # YAML null config test
    }
    callbacks = build_callbacks(configs, output_path=tmp_path)
    assert len(callbacks) == 5
    assert isinstance(callbacks[0], CheckpointCallBack)
    assert isinstance(callbacks[1], LoggingCallBack)
    assert isinstance(callbacks[2], EarlyStoppingCallBack)
    assert isinstance(callbacks[3], CSVLoggerCallBack)
    assert isinstance(callbacks[4], TimerCallBack)


def test_build_callbacks_none_or_disabled(tmp_path: Path) -> None:
    # 1. callback_configs is None
    cbs1 = build_callbacks(None, tmp_path)
    assert len(cbs1) == 0

    # 2. callback disabled with False
    cbs2 = build_callbacks({"logging": False, "timer": True}, tmp_path)
    assert len(cbs2) == 1
    assert isinstance(cbs2[0], TimerCallBack)
