#!/usr/bin/env python3
"""
We are going to implement the attention mechanism in the transformer architecture
"""

import torch
import torch.nn as nn
__author__ = "Mihir Deo"
__version__ = "0.1.0"
__license__ = "MIT"


class SelfAttention(nn.Module):
    """
    The key/value/query concepts come from retrieval systems. 
    For example, when you type a query to search for some video on Youtube, the search engine will map your query against a set of keys (video title, description etc.) associated with candidate videos in the database, then present you the best matched videos (values).
    """

    def __init__(self, input_dim, dropout_value=0.1) -> None:

        super(SelfAttention, self).__init__()

        self.input_dim = input_dim

        self.query_layer = nn.Linear(input_dim, input_dim)
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout_value)

    def scaled_dot_product_attention(self, query, key, value, mask):

        atten_scores = torch.matmul(query, key.transpose(-2, -1))

        atten_scores_scaled = torch.divide(
            atten_scores, torch.sqrt(torch.tensor(self.input_dim)))

        if mask is not None:
            atten_scores_scaled = atten_scores_scaled.masked_fill(
                mask == 0, -1e9)

        atten_prob = torch.softmax(atten_scores_scaled, dim=-1)

        output = torch.matmul(atten_prob, value)

        return output

    def forward(self, input_data, mask=None, encoder_output=None):

        query_w = self.query_layer(input_data)

        if encoder_output is not None:
            key_w = self.key_layer(encoder_output)
            value_w = self.value_layer(encoder_output)
        else:
            key_w = self.key_layer(input_data)
            value_w = self.value_layer(input_data)

        attention_results = self.scaled_dot_product_attention(
            query_w, key_w, value_w, mask)

        attention_results = self.dropout(attention_results)

        return attention_results

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, num_heads, input_dim) -> None:
        super(MultiHeadAttentionWrapper, self).__init__()

        self.heads = nn.ModuleList([SelfAttention(input_dim)
                                   for _ in range(num_heads)])

        self.out_proj = nn.Linear(input_dim*num_heads, input_dim)

    def forward(self, input_data, attention_mask=None, encoder_output=None):

        attention_scores = [head.forward(
            input_data, mask=attention_mask, encoder_output=encoder_output) for head in self.heads]

        context_vec = torch.cat(attention_scores, dim=-1)

        return self.out_proj(context_vec)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model * num_heads

        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"

        self.query_layer = nn.Linear(d_model, d_model * num_heads)
        self.key_layer = nn.Linear(d_model, d_model * num_heads)
        self.value_layer = nn.Linear(d_model, d_model * num_heads)

        self.proj_layer = nn.Linear(d_model * num_heads, d_model)

    def forward(self, input_data, mask=None,encoder_output=None):

        # Project Q, K, and V for each head
        query = self.query_layer(input_data).view(
            input_data.shape[0], input_data.shape[1], self.num_heads, self.d_model)
        
        if encoder_output:
            key = self.key_layer(encoder_output).view(
                encoder_output.shape[0], encoder_output.shape[1], self.num_heads, self.d_model)
            value = self.value_layer(encoder_output).view(
                encoder_output.shape[0], encoder_output.shape[1], self.num_heads, self.d_model)
        else:
            key = self.key_layer(input_data).view(
                input_data.shape[0], input_data.shape[1], self.num_heads, self.d_model)
            value = self.value_layer(input_data).view(
                input_data.shape[0], input_data.shape[1], self.num_heads, self.d_model)

        scores = torch.divide(torch.einsum(
            "nqhd,nkhd->nhqk", [query, key]), torch.sqrt(torch.tensor(self.d_model)))

        # Apply mask (if provided)
        if mask is not None:
            # Mask out padding values
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to scores (attention weights)
        weights = torch.softmax(scores, dim=-1)

        # Attention layer output
        output = torch.einsum("nhkq,nvhd->nvhd", weights, value)

        # Concatenate heads and reshape back
        concat = output.view(
            input_data.shape[0], input_data.shape[1], self.head_dim)

        # Final linear layer
        proj_layer = self.proj_layer(concat)

        return proj_layer

class MultiQueryAttentionWrapper(nn.Module):

    def __init__(self, num_queries, input_dim, dropout=0.1) -> None:
        super(MultiQueryAttentionWrapper, self).__init__()

        self.num_queries = num_queries
        self.input_dim = input_dim
        self.query_layer = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(num_queries)])
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.value_layer = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim*num_queries, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_data, mask=None, encoder_output=None):

        key_w = self.key_layer(input_data)
        value_w = self.value_layer(input_data)

        for query in self.query_layer:
            attention_src = torch.matmul(
                query(input_data), key_w.transpose(-2, -1))
            attention_src = attention_src / \
                torch.sqrt(torch.tensor(self.input_dim))
            attention_prob = torch.softmax(attention_src, dim=-1)
            attention_output = torch.matmul(attention_prob, value_w)
            attention_output = self.dropout(attention_output)
            attention_scores = torch.cat(attention_output, dim=-1)

        return self.out_proj(attention_scores)

class MultiQueryAttention(nn.Module):

    def __init__(self, d_model,num_queries) -> None:
        super(MultiQueryAttention, self).__init__()

        self.num_queries = num_queries
        self.d_model = d_model
        self.queries = self.num_queries * self.d_model

        assert d_model % num_queries == 0, "d_model must be a multiple of num_queries"

        self.query_layers = nn.Linear(d_model, d_model * num_queries)
        self.key_layer = nn.Linear(d_model, d_model)
        self.value_layer = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model*num_queries , d_model)

    def forward(self, input_data, mask=None):

        query = self.query_layers(input_data).view(input_data.shape[0],input_data.shape[1],self.num_queries,self.d_model)
        key = self.key_layer(input_data)
        value = self.value_layer(input_data)

        scores = torch.divide(torch.einsum("bqnd,bkd->bqnk",[query,key]),torch.sqrt(torch.tensor(self.d_model)))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        weights = torch.softmax(scores, dim=-1)

        output = torch.einsum("bqnk,bvd->bqnd",[weights,value])

        concat = output.view(input_data.shape[0],input_data.shape[1],self.queries)

        proj_layer = self.out_proj(concat)

        return proj_layer

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_groups, num_queries_per_group):
        super(GroupedQueryAttention, self).__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.num_queries_per_group = num_queries_per_group

        self.query_layers = nn.Linear(d_model, d_model * num_groups * num_queries_per_group)
        self.key_layer = nn.Linear(d_model, d_model*num_groups)
        self.value_layer = nn.Linear(d_model, d_model*num_groups)
        self.out_proj = nn.Linear(d_model*num_groups, d_model)

    def forward(self,input_data):
            
        query = self.query_layers(input_data).view(input_data.shape[0],input_data.shape[1],self.num_groups,self.num_queries_per_group,self.d_model)

        key = self.key_layer(input_data).view(input_data.shape[0],input_data.shape[1],self.num_groups,self.d_model)

        value = self.value_layer(input_data).view(input_data.shape[0],input_data.shape[1],self.num_groups,self.d_model)

        scores = torch.divide(torch.einsum("bqghd,bkgd-> bqhg", [query, key]), torch.sqrt(torch.tensor(self.d_model)))

        weights = torch.softmax(scores, dim=-1)

        output = torch.einsum("bqhg,bvgd->bvgd", [weights, value]).view(input_data.shape[0],input_data.shape[1],self.num_groups*self.d_model)

        proj_layer = self.out_proj(output)

        return proj_layer
