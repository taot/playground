import math
from typing import Optional, Callable

import torch
from torch import nn, Tensor
from torch.nn import LayerNorm
import numpy as np


def create_rotary_position_encoding_tensor(d_model: int, n_seq: int) -> Tensor:
    if d_model % 2 != 0:
        raise ValueError(f"d_model must be divisible by 2: d_model = {d_model}")

    R = np.zeros((n_seq, d_model, d_model), dtype=np.float32)
    for j in range(n_seq):
        for i in range(d_model // 2):
            theta = math.pow(10000, -2 * i / d_model)
            R[j, 2 * i, 2 * i] = math.cos(j * theta)
            R[j, 2 * i, 2 * i + 1] = -1 * math.sin(j * theta)
            R[j, 2 * i + 1, 2 * i] = math.sin(j * theta)
            R[j, 2 * i + 1, 2 * i + 1] = math.cos(j * theta)

    t = torch.from_numpy(R)
    t.requires_grad_(False)

    t = t.transpose(-1, -2)

    assert t.shape == torch.Size([n_seq, d_model, d_model])

    return t


def swap_adjacent_last_dim(x: Tensor) -> Tensor:
    last_dim = x.shape[-1]
    assert last_dim % 2 == 0

    # Create indices for swapping in the last dimension
    indices = torch.arange(last_dim, device=x.device)
    indices[0::2], indices[1::2] = indices[1::2].clone(), indices[0::2].clone()

    y = torch.gather(x, -1, indices.expand(x.shape))

    return y


def apply_rotary_position_encoding(x: Tensor, d: int, n_seq: int) -> Tensor:
    if d % 2 != 0:
        raise ValueError(f"d_model must be divisible by 2 in rotary position encoding: d_model = {d_model}")

    assert x.shape[-1] == d
    assert x.shape[-2] == n_seq

    x2 = swap_adjacent_last_dim(x)

    t_cos = torch.zeros([n_seq, d], dtype=torch.float32, requires_grad=False)
    t_sin = torch.zeros([n_seq, d], dtype=torch.float32, requires_grad=False)
    for j in range(n_seq):
        for i in range(d // 2):
            theta = math.pow(10000, -2 * i / d)
            t_cos[j, 2 * i] = math.cos(theta * j)
            t_cos[j, 2 * i + 1] = math.cos(theta * j)
            t_sin[j, 2 * i] = math.sin(theta * j)
            t_sin[j, 2 * i + 1] = math.sin(theta * j) * -1

    # this preserves gradients
    output = torch.mul(x, t_cos) + torch.mul(x2, t_sin)

    return output


class AttentionBlock(nn.Module):

    def __init__(self, d_model: int, *, weights: Optional[dict[str, Tensor]] = None) -> None:
        super().__init__()

        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        if weights is not None:
            self.W_q.weight = nn.Parameter(weights["W_q"])
            self.W_k.weight = nn.Parameter(weights["W_k"])
            self.W_v.weight = nn.Parameter(weights["W_v"])

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q, k, v: (batch_size, n_seq, d_model)
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == self.d_model
        assert q.shape == k.shape == v.shape

        # query, key, value: (batch_size, n_seq, d_model)
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        key_t = key.transpose(1, 2)                 # key_t: (batch_size, d_model, n_seq)
        scores = torch.matmul(query, key_t)         # scores: (batch_size, n_seq, n_seq)
        scaled_score = scores / math.sqrt(self.d_model)     # scale by 1 / sqrt(d_model) to avoid pushing softmax to regions where it has extremely small gradients
        softmax_scores = torch.softmax(scaled_score, dim=-1)      # weighted_scores: (batch_size, n_seq, n_seq), apply softmax for each query
        output = torch.matmul(softmax_scores, value)        # output: (batch_size, n_seq, d_mode), weighted sum on value for each query

        return output


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_seq: int, h: int, *, weights: Optional[dict[str, Tensor]] = None) -> None:
        super().__init__()

        if d_model % h != 0:
            raise ValueError(f"d_model must be divisible by h: d_model = {d_model}, h = {h}")

        self.d_model = d_model
        self.n_seq = n_seq
        self.h = h

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.linear = nn.Linear(d_model, d_model)

        if weights is not None:
            self.W_q.weight = nn.Parameter(weights["W_q"])
            self.W_k.weight = nn.Parameter(weights["W_k"])
            self.W_v.weight = nn.Parameter(weights["W_v"])
            self.linear.weight = nn.Parameter(weights["linear"])
            self.linear.bias = nn.Parameter(weights["linear_bias"])

    # TODO: add mask
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # q, k, v: (batch_size, n_seq, d_model)
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3
        assert q.shape[-1] == k.shape[-1] == v.shape[-1] == self.d_model
        assert q.shape == k.shape == v.shape

        # query, key, value: (batch_size, n_seq, d_model)
        query = self.W_q(q)
        key = self.W_k(k)
        value = self.W_v(v)

        # apply rotary position encoding
        query = apply_rotary_position_encoding(query, self.d_model, self.n_seq)
        key = apply_rotary_position_encoding(key, self.d_model, self.n_seq)

        d_k = self.d_model // self.h

        # query, key, value: (batch_size, h, n_seq, d_k)
        query = self._multi_head_transform(query, d_k)
        key = self._multi_head_transform(key, d_k)
        value = self._multi_head_transform(value, d_k)

        # key_t: (batch_size, h, d_k, n_seq)
        key_t = key.transpose(-1, -2)
        # scores, scaled_scores, softmax_scores: (batch_size, h, n_seq, n_seq)
        scores = torch.matmul(query, key_t)
        scaled_scores = scores / math.sqrt(d_k)
        softmax_scores = torch.softmax(scaled_scores, dim=-1)

        y1 = torch.matmul(softmax_scores, value)        # y1: (batch_size, h, n_seq, d_k)
        y2 = y1.transpose(1, 2)                         # y2: (batch_size, n_seq, h, d_k)
        y3 = y2.reshape(-1, self.n_seq, self.d_model)   # y3: (batch_size, n_seq, d_model)
        output = self.linear(y3)                        # output: (batch_size, n_seq, d_model)

        return output

    def _multi_head_transform(self, x: Tensor, d_k: int) -> Tensor:
        # x: (batch_size, n_seq, d_model)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.d_model
        assert x.shape[-2] == self.n_seq

        x1 = x.view(-1, self.n_seq, self.h, d_k)        # x1: (batch_size, n_seq, h, d_k)
        y = x1.transpose(1, 2)                          # y: (batch_size, h, n_seq, d_k)

        assert len(y.shape) == 4
        assert y.shape[1] == self.h
        assert y.shape[2] == self.n_seq
        assert y.shape[3] == d_k

        return y

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, *, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, n_seq, d_model)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.d_model

        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class ResidualConnection(nn.Module):

    def __init__(self, d_model: int, n_seq: int, *, dropout: float):
        super().__init__()

        self.d_model = d_model
        self.n_seq = n_seq
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(normalized_shape=[n_seq, d_model])

    def forward(self, x: Tensor, sub_layer: Callable[[Tensor], Tensor]) -> Tensor:
        # x: (batch_size, n_seq, d_model)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.d_model
        assert x.shape[-2] == self.n_seq

        x1 = sub_layer(x)
        x2 = self.dropout(x1)
        x3 = x + x2
        x4 = self.layer_norm(x3)

        return x4


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_seq: int, *, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.n_seq = n_seq

        self.attention = AttentionBlock(d_model)
        self.feed_forward = FeedForwardBlock(d_model, 2 * d_model, dropout=dropout)
        self.resid_conn_1 = ResidualConnection(d_model, n_seq, dropout=dropout)
        self.resid_conn_2 = ResidualConnection(d_model, n_seq, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, n_seq, d_model)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.d_model
        assert x.shape[-2] == self.n_seq

        x1 = self.resid_conn_1(x, lambda x: self.attention(x, x, x))
        x2 = self.resid_conn_2(x1, self.feed_forward)

        return x2


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, n_seq, d_model)
        assert len(x.shape) == 3
        assert x.shape[-1] == self.d_model

        x1 = self.linear(x)         # x1: (batch_size, n_seq, vocab_size)
        x2 = torch.softmax(x1, -1)  # x2: (batch_size, n_seq, vocab_size)

        return x2


class Transformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass