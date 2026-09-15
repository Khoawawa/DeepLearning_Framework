from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from torchvision.datasets import STL10, CIFAR10
from torchvision import transforms

from engines.registries import DATASET_REGISTRY

@DATASET_REGISTRY.register("stl10")
class STL10Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "unlabeled",
        image_size: int = 96,
        download: bool = True,
        **kwargs
    ) -> None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self._dataset = STL10(root=str(root), split=split, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _label = self._dataset[idx]  # discard label — JEPA doesn't need it
        return image

@DATASET_REGISTRY.register("cifar10")
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        image_size: int = 32,
        download: bool = True,
        **kwargs
    ) -> None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self._dataset = CIFAR10(root=str(root), train=train, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _label = self._dataset[idx]  # discard label — JEPA doesn't need it
        return image
    
@DATASET_REGISTRY.register("dummy")
class DummyImageDataset(Dataset):
    """Synthetic in-memory dataset for quickly testing trainers/callbacks
    without downloading anything. Returns a dict batch, matching
    BaseTrainer.fit()'s dict-based collate requirement.
    """

    def __init__(
        self,
        num_samples: int = 256,
        image_size: int = 32,
        num_channels: int = 3,
        seed: int | None = 0,
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self._images = torch.rand(
            num_samples, num_channels, image_size, image_size, generator=generator
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[idx]}
    