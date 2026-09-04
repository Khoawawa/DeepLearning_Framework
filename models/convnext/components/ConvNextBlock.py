from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
from functools import partial

from torchvision.ops.misc import Conv2dNormActivation, Permute 
from torchvision.ops.stochastic_depth import StochasticDepth
import yaml

@dataclass
class ConvNextBlockConfig:
    in_channels: int
    out_channels: Optional[int]
    num_layers: int
    
    @classmethod
    def from_yaml(cls, yaml_file_path: str) -> "ConvNextBlockConfig":
        
        with open(yaml_file_path, "r") as f:
            raw = yaml.safe_load(f)
        raw["block_settings"] = [ConvNextBlockConfig(**block) for block in raw["block_settings"]]
        return raw

class ConvNextBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]), # (B, C, H, W) -> (B, H, W, C)
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]), # (B, H, W, C) -> (B, C, H, W)
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, C, H, W)
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result