import torch
from typing import Tuple


class JepaMaskSampler:
    def __init__(
        self,
        num_targets: int = 4,
        target_scale: Tuple[float, float] = (0.15, 0.2),
        context_scale: Tuple[float, float] = (0.85, 1.0),
        aspect_ratio: Tuple[float, float] = (0.75, 1.5),
    ) -> None:
        self.num_targets = num_targets
        self.target_scale = target_scale
        self.context_scale = context_scale
        self.aspect_ratio = aspect_ratio

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, num_patches, _ = x.shape  # assumes x is already patchified: (B, N, D)
        device = x.device

        target_idx_list: list[torch.Tensor] = [
            self._sample_block_indices(batch_size, num_patches, self.target_scale, device)
            for _ in range(self.num_targets)
        ]

        # context = everything not covered by any target block
        context_idx = self._sample_context_indices(
            batch_size, num_patches, target_idx_list, device
        )

        return context_idx, target_idx_list

    def _sample_block_indices(
        self,
        batch_size: int,
        num_patches: int,
        scale_range: Tuple[float, float],
        device: torch.device,
    ) -> torch.Tensor:
        scale = torch.empty(1).uniform_(*scale_range).item()
        block_size = max(1, int(num_patches * scale))

        idx = torch.stack([
            torch.randperm(num_patches, device=device)[:block_size]
            for _ in range(batch_size)
        ])
        return idx  # (B, block_size)

    def _sample_context_indices(
        self,
        batch_size: int,
        num_patches: int,
        target_idx_list: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        scale = torch.empty(1).uniform_(*self.context_scale).item()
        context_size = max(1, int(num_patches * scale))

        # mask out target-covered patches per sample, then sample from the remainder
        context_idx = torch.stack([
            self._context_for_sample(b, num_patches, target_idx_list, context_size, device)
            for b in range(batch_size)
        ])
        return context_idx

    def _context_for_sample(
        self,
        b: int,
        num_patches: int,
        target_idx_list: list[torch.Tensor],
        context_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        covered = torch.cat([t[b] for t in target_idx_list])
        all_idx = torch.arange(num_patches, device=device)
        mask = torch.ones(num_patches, dtype=torch.bool, device=device)
        mask[covered] = False
        available = all_idx[mask]

        n = min(context_size, available.numel())
        perm = torch.randperm(available.numel(), device=device)[:n]
        return available[perm]