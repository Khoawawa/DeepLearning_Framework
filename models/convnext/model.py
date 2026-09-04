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
    def __init__(self, 
                 block_settings: list[ConvNextBlockConfig]
                 ) -> None:
        super().__init__()
        ...
        # norm_layer = partial(LayerNorm2D, eps=1e-6)
        
        # stem_out_channels: int = block_settings[0].in_channels  
        # self.stem = Conv2dNormActivation(
        #     in_channels=3,
        #     out_channels=stem_out_channels,
        #     kernel_size=4,
        #     stride=4,
        #     padding=0,
        #     norm_layer = norm_layer,
        #     activation_layer = None,
        #     bias=True
        # )
        
        # self.res2 = ConvNextBlock(
        #     stem_out_channels,
            
        # )