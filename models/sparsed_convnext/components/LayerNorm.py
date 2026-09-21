import torch
import torch.nn as nn

class LayerNorm2D(nn.LayerNorm):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        input = input.permute(0, 2, 3, 1)  # (B, H, W, C)
        input = super().forward(input)
        return input.permute(0, 3, 1, 2)  # (B, C, H, W)