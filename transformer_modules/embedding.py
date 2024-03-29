
#!/usr/bin/env python3
"""
We are going to implement the embedding layer related modules of the transformer architecture
"""

import torch
import torch.nn as nn
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, output_dim) -> None:
        super(EmbeddingLayer, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, output_dim)

    def forward(self, input_data):
        x = self.embeddings(input_data)
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, output_dim) -> None:
        super(PositionalEncoding, self).__init__()

        self.output_dim = output_dim
        self.pe = torch.zeros(seq_len, output_dim)

        self.position_index = torch.arange(0, seq_len).unsqueeze(1)

        self.pe[:, 0::2] = torch.sin(
            self.position_index / (10000 ** (self.position_index / self.output_dim)))

        self.pe[:, 1::2] = torch.cos(
            self.position_index / (10000 ** (self.position_index / self.output_dim)))

    def forward(self, input_data):

        input_data = input_data + self.pe[:input_data.size(1), :].unsqueeze(0)

        return input_data