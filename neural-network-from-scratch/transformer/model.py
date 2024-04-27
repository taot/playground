import torch
from torch import nn, Tensor
import math
from typing import Optional, Callable

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len), dtype=long
        x = self.embedding(x) * math.sqrt(self.d_model) # (B, seq_len, d_model)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, *, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, step=2, dtype=torch.float) * (-math.log(10000) / d_model))    # (d_model)

        p = position * div_term

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))    # multiplied
        self.bias = nn.Parameter(torch.zeros(1))    # added

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        y = ((x - mean) / (std + self.eps)) * self.alpha + self.bias
        return y

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, *, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    # TODO: experiment ReLU approach in the paper
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, d_model)
        # (B, seq_len, d_model) -> (B, seq_len, d_ff) -> (B, seq_len, d_model)
        x = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        return x

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, *, dropout: float) -> None:
        super().__init__()

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_model = d_model
        self.h = h

        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor], dropout: Optional[nn.Dropout]) -> (Tensor, Tensor):
        d_k = query.size(-1)

        attention_scores = query @ key.transpose(-2, -1)  # (B, h, seq_len, seq_len)
        if mask is not None:
            # why not mask after sofmax?
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores   # (B, h, seq_len, d_k), (B, h, seq_len, seq_len)


    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        d_k = self.d_model // self.h

        # (B, seq_len, d_model) -> (B, seq_len, h, d_k) -> (B, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.h, d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, d_k).transpose(1, 2)

        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (B, h, seq_len, d_k) -> (B, seq_len, h, d_k) -> (B, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)

        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, *, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: Tensor, sublayer: Callable[Tensor, Tensor]) -> Tensor:
        x = x + self.dropout(sublayer(self.norm(x)))

        # it's different in the paper
        # x = self.norm(self.dropout(x + sublayer(x)))

        return x

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feedforward_block: FeedForwardBlock, *, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(2)])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feedforward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
        feedforward_block: FeedForwardBlock, *, dropout: float):

        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout=dropout) for _ in range(3)])
        
    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        self.residual_connections[2](x, self.feedforward_block)
        return x

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        x = torch.log_softmax(self.proj(x), dim=-1)
        return x

