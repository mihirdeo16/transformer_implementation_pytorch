#!/usr/bin/env python3
"""
Wrapper for feedforward network
"""

import torch
import torch.nn as nn
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


class FeedForward(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=1024) -> None:
        super(FeedForward, self).__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, input_data):

        output = self.act(self.fc_1(input_data))

        return self.act(self.fc_2(output))