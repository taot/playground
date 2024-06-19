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
        self.dropout = nn.Dropout(dropout)  # why positional encoding needs a dropout?

        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, step=2, dtype=torch.float) * (-math.log(10000) / d_model))    # (d_model)

        # p = position * div_term

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))    # multiplied
        self.bias = nn.Parameter(torch.zeros(features))    # added

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

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, h, seq_len, seq_len)
        
        if mask is not None:
            # why not mask after softmax?
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores   # (B, h, seq_len, d_k), (B, h, seq_len, seq_len)

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

    def __init__(self, features: int, *, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        x = x + self.dropout(sublayer(self.norm(x)))

        # it's different in the paper
        # x = self.norm(self.dropout(x + sublayer(x)))

        return x


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, *, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout=dropout) for _ in range(2)])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, *, dropout: float):

        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout=dropout) for _ in range(3)])
        
    def forward(self, x: Tensor, encoder_output: Tensor, cross_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, cross_mask))    # TODO
        self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)
    
    def forward(self, x: Tensor, encoder_output: Tensor, cross_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, cross_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # (B, seq_len, d_model) -> (B, seq_len, vocab_size)
        # x = torch.log_softmax(self.proj(x), dim=-1)
        x = self.proj(x)
        return x


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        x = self.src_embed(src) # (B, seq_len) -> (B, seq_len, d_model)
        x = self.src_pos(x)     # (B, seq_len, d_model) -> (B, seq_len, d_model)
        x = self.encoder(x, src_mask)   # (B, seq_len, d_model) -> (B, seq_len, d_model)
        return x

    def decode(self, encoder_output: Tensor, cross_mask: Tensor, tgt: Tensor, tgt_mask: Tensor) -> Tensor:
        x = self.tgt_embed(tgt)     # (B, seq_len) -> (B, seq_len, d_model)
        x = self.tgt_pos(x)         # (B, seq_len, d_model) -> (B, seq_len, d_model)
        x = self.decoder(x, encoder_output, cross_mask, tgt_mask)
        return x    # (B, seq_len, d_model)

    def project(self, x: Tensor) -> Tensor:
        return self.projection_layer(x)     # (B, seq_len, d_model) -> (B, seq_len, vocab_size)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, N: int = 6, h: int = 8,  dropout: float = 0.1, d_ff: int = 2048) -> Transformer:

    # Embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Positional encodings
    src_pos_encoding = PositionalEncoding(d_model, src_seq_len, dropout=dropout)
    tgt_pos_encoding = PositionalEncoding(d_model, tgt_seq_len, dropout=dropout)

    # Encoder blocks
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout=dropout)
        encoder_block = EncoderBlock(d_model, self_attention_block, feed_forward_block, dropout=dropout)
        encoder_blocks.append(encoder_block)

    # Decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout=dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout=dropout)
        decoder_block = DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout=dropout)
        decoder_blocks.append(decoder_block)
    
    # Encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Build transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos_encoding, tgt_pos_encoding, projection_layer)

    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
