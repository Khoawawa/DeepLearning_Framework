from pathlib import Path
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader, Dataset
from data_modules.dataset import CIFAR10Dataset
from engines.factory import Factory, DATASET_REGISTRY


def test_cifar10_dataset_registration() -> None:
    assert "cifar10" in DATASET_REGISTRY
    assert DATASET_REGISTRY.get("cifar10") == CIFAR10Dataset


@patch("data_modules.dataset.CIFAR10")
def test_cifar10_dataset_init_and_methods(mock_cifar10: MagicMock, tmp_path: Path) -> None:
    # Create mock underlying CIFAR10 data
    mock_instance = MagicMock()
    mock_instance.__len__.return_value = 100
    mock_tensor = torch.zeros((3, 32, 32))
    mock_instance.__getitem__.return_value = (mock_tensor, 3)  # image, label
    mock_cifar10.return_value = mock_instance

    dataset = CIFAR10Dataset(root=tmp_path, train=True, image_size=32, download=False)
    
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 100
    
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 32, 32)


@patch("data_modules.dataset.CIFAR10")
def test_factory_build_cifar10(mock_cifar10: MagicMock, tmp_path: Path) -> None:
    mock_instance = MagicMock()
    mock_instance.__len__.return_value = 50
    mock_tensor = torch.zeros((3, 32, 32))
    mock_instance.__getitem__.return_value = (mock_tensor, 0)
    mock_cifar10.return_value = mock_instance

    factory = Factory()
    data_cfg = {
        "name": "cifar10",
        "root": str(tmp_path),
        "batch_size": 4,
        "num_workers": 0,
        "download": False
    }

    built_dataset = factory.build_dataset(data_cfg)
    assert isinstance(built_dataset, CIFAR10Dataset)
    assert len(built_dataset) == 50

    dataloader = factory.build_dataloader(data_cfg, is_train=True)
    assert isinstance(dataloader, DataLoader)
