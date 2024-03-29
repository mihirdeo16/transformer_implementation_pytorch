# Transformer implementation from scratch in PyTorch

This repository contains an implementation of the Transformer architecture from scratch using PyTorch. It includes implementations of various components, including the embedding layer, positional encoding, multi-head attention (masked and unmasked), Multi Query Attention (MQA), Group Query Attention (GQA), Add & Norm, and feedforward layers.


### Implemented Modules (More to Come! 🚀)
+ Embedding Layer: Implements the embedding layer for converting input tokens to vectors.
+ Positional Encoding: Implements positional encoding using sine and cosine functions to provide positional information to the model.
+ Self-Attention: Base layer/form of attention mechanism 
+ Multi-Head Attention: Implements multi-head attention mechanism, both masked and unmasked, for capturing relationships between different words in the input.
+ Multi Query Attention (MQA): Implements Multi Query Attention for capturing multiple query vectors in the attention mechanism.
+ Group Query Attention (GQA): Implements Group Query Attention for grouping query vectors in the attention mechanism.
+ Normalization: Implements the "Add & Norm" layer for adding skip connections and applying layer normalization.
+ Feed Forward: Implements the feedforward layer with activation functions for non-linear transformations.


#### License
This project is licensed under the MIT License - see the LICENSE file for details.