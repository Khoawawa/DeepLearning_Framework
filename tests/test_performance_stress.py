import gc
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tests.dummy_datasets import DummyImageDataset
from engines.callbacks import LoggingCallBack, TimerCallBack
from engines.trainer import CNNTrainer


@pytest.mark.performance
def test_large_dataset_throughput_performance(tmp_path: Path) -> None:
    """Stress test system throughput under a heavy synthetic dataset (50,000 samples).

    Verifies processing rate (samples/sec) and system stability.
    """
    num_samples = 50000
    batch_size = 512
    image_size = 16

    dataset = DummyImageDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_channels=3,
        label_shape=(16, 14, 14),
        seed=42,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    cnn_block = nn.Conv2d(3, 16, kernel_size=3)
    opt = torch.optim.Adam(cnn_block.parameters(), lr=0.001)
    crit = nn.MSELoss()

    trainer = CNNTrainer(cnn_block=cnn_block, optimizer=opt, criterion=crit)
    cb_logging = LoggingCallBack(output_path=tmp_path, log_every_steps=50)

    start_time = time.perf_counter()
    trainer.fit(dataloader, num_epochs=1, call_backs=[cb_logging])
    elapsed = time.perf_counter() - start_time

    samples_per_sec = num_samples / max(elapsed, 1e-6)
    assert trainer.global_step == len(dataloader)
    assert elapsed > 0
    assert samples_per_sec > 1000, f"Throughput too low: {samples_per_sec:.2f} samples/sec"


@pytest.mark.performance
def test_high_resolution_heavy_tensor_stress(tmp_path: Path) -> None:
    """Stress test system under high-resolution image tensors (256x256) and larger layer channels.

    Verifies memory allocation and gradient computation stability under heavy tensor shapes.
    """
    num_samples = 256
    batch_size = 16
    image_size = 256  # High resolution input image

    # Conv block out shape for 256x256 with 7x7 conv kernel, stride=1, pad=0 -> (64, 250, 250)
    dataset = DummyImageDataset(
        num_samples=num_samples,
        image_size=image_size,
        num_channels=3,
        label_shape=(64, 250, 250),
        seed=100,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    heavy_cnn = nn.Conv2d(3, 64, kernel_size=7)
    opt = torch.optim.AdamW(heavy_cnn.parameters(), lr=0.0005)
    crit = nn.MSELoss()

    trainer = CNNTrainer(cnn_block=heavy_cnn, optimizer=opt, criterion=crit)
    cb_timer = TimerCallBack(output_path=tmp_path, log_every_steps=5)

    trainer.fit(dataloader, num_epochs=2, call_backs=[cb_timer])
    assert trainer.current_epoch == 2
    assert trainer.global_step == (num_samples // batch_size) * 2


@pytest.mark.performance
def test_multi_epoch_memory_leak_stability(tmp_path: Path) -> None:
    """Stress test memory stability across 10 continuous training epochs under dataset load.

    Ensures callback state buffers, metric dicts, and tensor graphs do not cause memory inflation.
    """
    dataset = DummyImageDataset(
        num_samples=1000,
        image_size=32,
        num_channels=3,
        label_shape=(16, 30, 30),
    )
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    cnn_block = nn.Conv2d(3, 16, kernel_size=3)
    opt = torch.optim.SGD(cnn_block.parameters(), lr=0.01)
    crit = nn.MSELoss()

    trainer = CNNTrainer(cnn_block=cnn_block, optimizer=opt, criterion=crit)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
    else:
        mem_before = 0

    trainer.fit(dataloader, num_epochs=10)

    gc.collect()
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated()
        # Ensure memory inflation stays within reasonable bound (< 50MB)
        assert (mem_after - mem_before) < 50 * 1024 * 1024

    assert trainer.current_epoch == 10
    assert trainer.global_step == 100
