#!/usr/bin/env python3
"""
Transformer architecture
"""

import torch
import torch.nn as nn
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"

"""
Transformer architecture is made up of following components:
+ Embedding layer
+ Positional encoding
+ Multi-head attention (Masked/Unmasked)
+ Add & Norm
+ Feed forward
"""

from transformer_modules.embedding import EmbeddingLayer, PositionalEncoding
from transformer_modules.attention import MultiheadAttention
from transformer_modules.feedforward_network import FeedForward
from transformer_modules.normalization_layer import Normalization


class EncoderLayer(nn.Module):

    def __init__(self, seq_len=512, d_model=768, num_head=10, vocab_size=500000) -> None:
        super(EncoderLayer, self).__init__()

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size, output_dim=d_model)

        self.position_encoding = PositionalEncoding(
            seq_len=seq_len, output_dim=d_model)

        self.multi_head_attention = MultiheadAttention(
            num_heads=num_head, d_model=d_model)

        self.normalization_layer_1 = Normalization(input_dim=d_model)

        self.feedforward_layer = FeedForward(
            input_dim=d_model, output_dim=d_model)

        self.normalization_layer_2 = Normalization(input_dim=d_model)

    def forward(self, input_data, attention_mask=None):

        out_1 = self.embedding_layer.forward(input_data)
        out_2 = self.position_encoding.forward(out_1)

        out_3 = self.multi_head_attention.forward(
            out_2, mask=attention_mask)

        out_4 = self.normalization_layer_1.forward(out_2+out_3)

        out_5 = self.feedforward_layer.forward(out_4)

        out_6 = self.normalization_layer_2.forward(out_4+out_5)

        return out_6

class DecoderLayer(nn.Module):
    def __init__(self, seq_len=512, d_model=768, num_head=10, vocab_size=500000):
        super(DecoderLayer, self).__init__()

        self.embedding_layer = EmbeddingLayer(
            vocab_size=vocab_size, output_dim=d_model)

        self.position_encoding = PositionalEncoding(
            seq_len=seq_len, output_dim=d_model)

        self.cross_multi_head_attention = MultiheadAttention(
            num_heads=num_head, d_model=d_model)
        
        self.masked_multi_head_attention = MultiheadAttention(
            num_heads=num_head, d_model=d_model)

        self.normalization_layer_1 = Normalization(input_dim=d_model)

        self.normalization_layer_2 = Normalization(input_dim=d_model)

        self.normalization_layer_3 = Normalization(input_dim=d_model)

        self.feedforward_layer = FeedForward(
            input_dim=d_model, output_dim=d_model)

    def forward(self,input_data,masked_attention,cross_attention,encoder_output):

        out_1 = self.embedding_layer.forward(input_data)
        out_2 = self.position_encoding.forward(out_1)

        out_3 = self.masked_multi_head_attention.forward(
            out_2, mask=masked_attention)

        out_4 = self.normalization_layer_1.forward(out_2+out_3)

        out_5 = self.cross_multi_head_attention.forward(
            out_2, mask=cross_attention,encoder_output=encoder_output)

        out_6 = self.normalization_layer_2.forward(out_4+out_5)

        out_7 = self.feedforward_layer.forward(out_6)

        out_8 = self.normalization_layer_2.forward(out_6+out_7)

        return out_8

class Transformer(nn.Module):

    def __init__(self, seq_len=512, d_model=768, num_class=10, num_head=10, vocab_size=500000) -> None:
        super(Transformer, self).__init__()

        self.encoder = EncoderLayer(seq_len=512, d_model=768, num_head=10, vocab_size=500000)

        self.decoder = DecoderLayer(seq_len=512, d_model=768, num_head=10, vocab_size=500000)

        self.output_layer = nn.Linear(d_model,num_class)

    def forward(self,src_input,src_mask,tgt_input,tgt_mask,cross_attention=None):

        encoder_output = self.encoder.forward(input_data=src_input,attention_mask=src_mask)

        decoder_output = self.decoder.forward(input_data=tgt_input,masked_attention=tgt_mask,cross_attention=cross_attention, encoder_output = encoder_output)

        output = self.output_layer(decoder_output)

        # Apply softmax to get the class probabilities
        output = torch.softmax(output,dim=-1)

        return output