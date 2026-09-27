from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_modules.dataset import DummyImageDataset
from engines.factory import Factory
from engines.tester import BaseTester, CNNTester
from engines.trainer import BaseTrainer, CNNTrainer
from engines.interfaces.irunner import CallBack, ITrainer, ITester


def test_factory_build_dataset_and_dataloader() -> None:
    factory = Factory()
    data_cfg = {
        "name": "dummy",
        "num_samples": 64,
        "image_size": 16,
        "batch_size": 8,
        "num_workers": 0,
        "label_shape": (16, 14, 14),
    }

    dataset = factory.build_dataset(data_cfg)
    assert isinstance(dataset, DummyImageDataset)
    assert len(dataset) == 64

    loader = factory.build_dataloader(data_cfg, is_train=True)
    assert isinstance(loader, DataLoader)
    assert len(loader) == 8


def test_factory_build_dataset_missing_name() -> None:
    factory = Factory()
    with pytest.raises(ValueError, match="Dataset must have a `name`/`type` field"):
        factory.build_dataset({"batch_size": 4})


def test_factory_build_trainer() -> None:
    factory = Factory()
    cfg = {
        "name": "cnn",
        "cnn_block": {"in_channels": 3, "out_channels": 16, "kernel_size": 3},
        "optimizers": {
            "optimizer": {"type": "Adam", "lr": 0.001}
        },
        "criteria": {
            "main": {"type": "mse"}
        },
    }

    trainer = factory.build_trainer(cfg)
    assert isinstance(trainer, CNNTrainer)
    assert isinstance(trainer, ITrainer)


def test_factory_build_trainer_missing_component() -> None:
    factory = Factory()
    cfg = {"name": "cnn"}  # missing cnn_block, optimizers, criteria
    with pytest.raises(ValueError, match="requires component"):
        factory.build_trainer(cfg)


def test_factory_build_tester() -> None:
    factory = Factory()
    cfg = {
        "name": "cnn",
        "cnn_block": {"in_channels": 3, "out_channels": 16, "kernel_size": 3},
        "criteria": {
            "main": {"type": "mse"}
        }
    }

    tester = factory.build_tester(cfg)
    assert isinstance(tester, CNNTester)
    assert isinstance(tester, ITester)


def test_factory_build_tester_missing_required() -> None:
    factory = Factory()
    with pytest.raises(ValueError, match="requires config field"):
        factory.build_tester({"name": "cnn"})


def test_cnn_trainer_fit_lifecycle(tmp_path: Path) -> None:
    cnn_block = nn.Conv2d(3, 16, kernel_size=3)
    opt = torch.optim.SGD(cnn_block.parameters(), lr=0.01)
    crit = nn.MSELoss()

    trainer = CNNTrainer(cnn_block=cnn_block, optimizer=opt, criterion=crit)
    
    # Dummy data with input & label matching output shape (16, 14, 14)
    dataset = DummyImageDataset(num_samples=16, image_size=16, label_shape=(16, 14, 14))
    loader = DataLoader(dataset, batch_size=4)

    mock_cb = MagicMock(spec=CallBack)
    trainer.fit(loader, num_epochs=2, call_backs=[mock_cb])

    assert trainer.current_epoch == 2
    assert trainer.global_step == 8  # 4 batches * 2 epochs
    assert mock_cb.on_step_end.call_count == 8
    assert mock_cb.on_epoch_end.call_count == 2
    assert mock_cb.on_training_end.call_count == 1


def test_cnn_trainer_resume_checkpoint(tmp_path: Path) -> None:
    cnn_block = nn.Conv2d(3, 16, kernel_size=3)
    opt = torch.optim.SGD(cnn_block.parameters(), lr=0.01)
    crit = nn.MSELoss()

    trainer = CNNTrainer(cnn_block=cnn_block, optimizer=opt, criterion=crit)
    trainer.current_epoch = 5
    trainer.global_step = 100

    ckpt_path = tmp_path / "test_ckpt.pt"
    torch.save(trainer.state_dict(), ckpt_path)

    # Instantiate fresh trainer and resume
    new_trainer = CNNTrainer(
        cnn_block=nn.Conv2d(3, 16, kernel_size=3),
        optimizer=torch.optim.SGD(nn.Conv2d(3, 16, 3).parameters(), lr=0.01),
        criterion=nn.MSELoss(),
    )

    dataset = DummyImageDataset(num_samples=8, image_size=16, label_shape=(16, 14, 14))
    loader = DataLoader(dataset, batch_size=4)

    new_trainer.fit(loader, num_epochs=6, resume_path=ckpt_path)
    assert new_trainer.current_epoch == 6
    assert new_trainer.global_step == 102


def test_cnn_trainer_resume_non_existent_path(tmp_path: Path) -> None:
    trainer = CNNTrainer(
        cnn_block=nn.Conv2d(3, 16, 3),
        optimizer=torch.optim.SGD(nn.Conv2d(3, 16, 3).parameters(), lr=0.01),
        criterion=nn.MSELoss(),
    )
    dataset = DummyImageDataset(num_samples=4)
    loader = DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="does not exist"):
        trainer.fit(loader, num_epochs=1, resume_path=tmp_path / "missing.pt")


def test_cnn_trainer_unsupported_checkpoint_type(tmp_path: Path) -> None:
    trainer = CNNTrainer(
        cnn_block=nn.Conv2d(3, 16, 3),
        optimizer=torch.optim.SGD(nn.Conv2d(3, 16, 3).parameters(), lr=0.01),
        criterion=nn.MSELoss(),
    )
    fake_ckpt = tmp_path / "ckpt.invalid"
    fake_ckpt.write_text("data", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported checkpoint file type"):
        trainer._load_from_checkpoint(fake_ckpt)


def test_cnn_tester_test_lifecycle(tmp_path: Path) -> None:
    tester = CNNTester(cnn_block=nn.Conv2d(3, 16, kernel_size=3), criterion=nn.MSELoss())
    
    # Dataset with label matching output shape
    dataset = DummyImageDataset(num_samples=12, image_size=16, label_shape=(16, 14, 14))
    loader = DataLoader(dataset, batch_size=4)

    mock_cb = MagicMock(spec=CallBack)
    metrics = tester.test(loader, call_backs=[mock_cb])

    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)
    assert tester.global_step == 3
    assert mock_cb.on_step_end.call_count == 3
    assert mock_cb.on_epoch_end.call_count == 1
    assert mock_cb.on_training_end.call_count == 1


def test_cnn_tester_test_without_labels() -> None:
    tester = CNNTester(cnn_block=nn.Conv2d(3, 16, kernel_size=3))
    
    # Tensor dataset without labels
    x = torch.randn(8, 3, 16, 16)
    loader = DataLoader(TensorDataset(x), batch_size=4)

    metrics = tester.test(loader)
    assert "output_mean" in metrics
    assert isinstance(metrics["output_mean"], float)


def test_tester_load_weights_file_not_found(tmp_path: Path) -> None:
    tester = CNNTester(cnn_block=nn.Conv2d(3, 16, 3))
    with pytest.raises(FileNotFoundError, match="not found"):
        tester.load_weights(tmp_path / "missing_weight.pt")


def test_tester_unpack_batch_invalid_type() -> None:
    tester = CNNTester(cnn_block=nn.Conv2d(3, 16, 3))
    with pytest.raises(ValueError, match="Input must be a torch.Tensor"):
        tester._unpack_batch(["not_a_tensor"])


def test_tester_infer_batch_size_invalid() -> None:
    tester = CNNTester(cnn_block=nn.Conv2d(3, 16, 3))
    with pytest.raises(ValueError, match="Cannot infer batch size"):
        tester._infer_batch_size(12345)


def test_tester_empty_dataloader_returns_empty_metrics() -> None:
    tester = CNNTester(cnn_block=nn.Conv2d(3, 16, 3))
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 16, 16)), batch_size=4)
    metrics = tester.test(empty_loader)
    assert metrics == {}
