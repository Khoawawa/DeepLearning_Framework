import torch
from torch.utils.data import Dataset
from torchvision.datasets import STL10
from torchvision import transforms

class STL10Dataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "unlabeled",
        image_size: int = 96,
        download: bool = True,
        **kwargs
    ) -> None:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self._dataset = STL10(root=root, split=split, download=download, transform=transform)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _label = self._dataset[idx]  # discard label — JEPA doesn't need it
        return image