from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from data_modules.dataset import CIFAR10Dataset, DummyImageDataset, STL10Dataset
from main import configure_logging, resolve_profile
from models.common import JepaMaskSampler
from utils.torch_utils import resolve_device, set_seed
from engines.engine_utils import to_var


def test_dummy_image_dataset_shapes_and_reproducibility() -> None:
    dataset1 = DummyImageDataset(
        num_samples=20,
        image_size=32,
        num_channels=3,
        num_classes=5,
        seed=123,
    )
    assert len(dataset1) == 20
    sample1 = dataset1[0]
    assert "image" in sample1 and "label" in sample1
    assert sample1["image"].shape == (3, 32, 32)
    assert sample1["label"].dtype in (torch.int64, torch.int32)

    # Label shape specified mode (e.g. for Conv block outputs)
    dataset2 = DummyImageDataset(
        num_samples=10,
        image_size=32,
        num_channels=3,
        label_shape=(16, 30, 30),
        seed=123,
    )
    sample2 = dataset2[0]
    assert sample2["label"].shape == (16, 30, 30)


@patch("data_modules.dataset.STL10")
def test_stl10_dataset_init_and_getitem(mock_stl10: MagicMock, tmp_path: Path) -> None:
    mock_instance = MagicMock()
    mock_instance.__len__.return_value = 30
    mock_tensor = torch.zeros((3, 96, 96))
    mock_instance.__getitem__.return_value = (mock_tensor, 0)
    mock_stl10.return_value = mock_instance

    dataset = STL10Dataset(root=tmp_path, split="unlabeled", image_size=96, download=False)
    assert len(dataset) == 30

    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 96, 96)


def test_jepa_mask_sampler() -> None:
    sampler = JepaMaskSampler(
        num_targets=2,
        target_scale=(0.1, 0.2),
        context_scale=(0.5, 0.8),
    )
    # Batch of 4 samples, 100 patches each, 64 dimension
    x = torch.randn(4, 100, 64)
    ctx_idx, target_idx_list = sampler.sample(x)

    assert len(target_idx_list) == 2
    assert ctx_idx.shape[0] == 4
    assert ctx_idx.ndim == 2


def test_torch_utils_and_engine_utils() -> None:
    set_seed(42)

    # Test device resolution
    dev_cpu = resolve_device("cpu")
    assert dev_cpu.type == "cpu"

    dev_auto = resolve_device(None)
    assert dev_auto.type in ("cpu", "cuda")

    # Test to_var recursively for dict, list, tuple, tensor
    batch = {
        "x": torch.randn(2, 3),
        "nested": [torch.randn(2), torch.randn(4)],
        "num": 42,
    }
    converted = to_var(batch, dev_cpu)
    assert isinstance(converted["x"], torch.Tensor)
    assert converted["x"].device == dev_cpu
    assert isinstance(converted["nested"][1], torch.Tensor)
    assert converted["num"] == 42


def test_logging_configuration_profiles(tmp_path: Path) -> None:
    profile_dev = resolve_profile()
    assert profile_dev in ("dev", "prod")

    configure_logging(log_dir=tmp_path / "logs", profile="dev")
    configure_logging(log_dir=tmp_path / "logs", profile="prod")
    assert (tmp_path / "logs").exists()
