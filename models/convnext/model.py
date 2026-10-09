from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch
import timm
import torchvision
from torchvision.ops import Conv2dNormActivation
from models.convnext.components.ConvNextBlock import ConvNextBlock, ConvNextBlockConfig
from typing import Callable, Optional

from functools import partial
from models.convnext.components.LayerNorm import LayerNorm2D

@dataclass
class ConvNextConfig:
    in_channels: int
    out_channels: int

class ConvNext(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depths: list[int] | None = None,
        dims: list[int] | None = None,
        drop_path_rate: float = 0.0,
        block_settings: list[ConvNextBlockConfig] | None = None,
    ) -> None:
        super().__init__()
        dims = dims if dims is not None else [64, 128, 256, 512]
        depths = depths if depths is not None else [2, 2, 6, 2]

        if block_settings is not None and len(block_settings) > 0:
            dims = [b.in_channels for b in block_settings]
            depths = [b.num_layers for b in block_settings]

        self.stem = Conv2dNormActivation(
            in_channels=in_channels,
            out_channels=dims[0],
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=LayerNorm2D,
            activation_layer=None,
            bias=True,
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(len(dims)):
            if i > 0 and dims[i] != dims[i - 1]:
                downsample = nn.Sequential(
                    LayerNorm2D(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                )
            else:
                downsample = nn.Identity()
            self.downsamples.append(downsample)

            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    ConvNextBlock(
                        dim=dims[i],
                        stochastic_depth_prob=dp_rates[cur + j],
                    )
                )
            cur += depths[i]
            self.stages.append(nn.Sequential(*stage_blocks))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for downsample, stage in zip(self.downsamples, self.stages):
            x = downsample(x)
            x = stage(x)
        return self.head(x)
