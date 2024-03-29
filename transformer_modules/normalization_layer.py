#!/usr/bin/env python3
"""
Wrapper for normalization layer
"""

import torch
import torch.nn as nn
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


class Normalization(nn.Module):

    def __init__(self, input_dim) -> None:
        super(Normalization, self).__init__()

        self.norm_layer = nn.LayerNorm(input_dim)

    def forward(self, input_data):
        return self.norm_layer(input_data)