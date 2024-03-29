
#!/usr/bin/env python3
"""
Basic BERT architecture
"""

from transformer_modules.normalization_layer import Normalization
from transformer_modules.feedforward_network import FeedForward
from transformer_modules.attention import SelfAttention
from transformer_modules.embedding import EmbeddingLayer, PositionalEncoding
import torch.nn as nn

__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


class BERT(nn.Module):

    def __init__(self, bert_layer=5, seq_len=512, d_model=768, vocab_size=500000) -> None:
        super(BERT, self).__init__()

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size, output_dim=d_model)
        self.positional_encoding = PositionalEncoding(
            seq_len=seq_len, output_dim=d_model)

        self.bert_layer = nn.ModuleList([
            nn.Sequential(
                SelfAttention(input_dim=d_model,),
                FeedForward(input_dim=d_model, output_dim=d_model),
                Normalization(input_dim=d_model),
                FeedForward(input_dim=d_model, output_dim=d_model),
            )
            for _ in range(bert_layer)])

    def forward(self, input_data, attention_mask=None):

        out_1 = self.embedding_layer(input_data)
        out_2 = self.positional_encoding(out_1)

        out_3 = self.bert_layer(
            out_2, mask=attention_mask)

        return out_3


if __name__ == "__main__":
    model = BERT()
    print(model)
