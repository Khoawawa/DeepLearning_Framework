from typing import Any
import torch
from torch.utils.data import Dataset
from engines.registries import DATASET_REGISTRY


@DATASET_REGISTRY.register("dummy")
class DummyImageDataset(Dataset):
    """Synthetic in-memory image dataset for testing trainers/callbacks."""

    def __init__(
        self,
        num_samples: int = 256,
        image_size: int = 32,
        num_channels: int = 3,
        num_classes: int = 10,
        label_shape: tuple[int, ...] | None = None,
        seed: int | None = 0,
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.label_shape = label_shape

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self._images = torch.rand(num_samples, num_channels, image_size, image_size, generator=generator)

        if label_shape is not None:
            self._labels = torch.rand(num_samples, *label_shape, generator=generator)
        else:
            self._labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[idx], "label": self._labels[idx]}


@DATASET_REGISTRY.register("dummy_text")
class DummyTextDataset(Dataset):
    """Synthetic in-memory dataset for text/nlp testing."""

    def __init__(
        self,
        num_samples: int = 128,
        vocab_size: int = 500,
        seq_len: int = 32,
        num_classes: int = 5,
        seed: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self._tokens = torch.randint(0, vocab_size, (num_samples, seq_len), generator=generator)
        self._labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"text": self._tokens[idx], "label": self._labels[idx]}


@DATASET_REGISTRY.register("dummy_audio")
class DummyAudioDataset(Dataset):
    """Synthetic in-memory dataset for audio testing."""

    def __init__(
        self,
        num_samples: int = 128,
        in_channels: int = 1,
        freq_bins: int = 64,
        time_steps: int = 64,
        num_classes: int = 10,
        seed: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self._audio = torch.randn(num_samples, in_channels, freq_bins, time_steps, generator=generator)
        self._labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"audio": self._audio[idx], "label": self._labels[idx]}


@DATASET_REGISTRY.register("dummy_multimodal")
class DummyMultimodalDataset(Dataset):
    """Synthetic in-memory dataset combining image, text, and audio inputs."""

    def __init__(
        self,
        num_samples: int = 128,
        image_size: int = 32,
        vocab_size: int = 500,
        seq_len: int = 16,
        audio_bins: int = 32,
        audio_steps: int = 32,
        num_classes: int = 10,
        seed: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.num_samples = num_samples
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        self._images = torch.rand(num_samples, 3, image_size, image_size, generator=generator)
        self._tokens = torch.randint(0, vocab_size, (num_samples, seq_len), generator=generator)
        self._audio = torch.randn(num_samples, 1, audio_bins, audio_steps, generator=generator)
        self._labels = torch.randint(0, num_classes, (num_samples,), generator=generator)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": self._images[idx],
            "text": self._tokens[idx],
            "audio": self._audio[idx],
            "label": self._labels[idx],
        }
