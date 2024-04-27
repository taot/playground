import torch
from torch import nn

from model import InputEmbeddings, PositionalEncoding, LayerNormalization, FeedForwardBlock, MultiHeadAttentionBlock, ResidualConnection
from model import EncoderBlock, Encoder, DecoderBlock, Decoder, ProjectionLayer, Transformer

B = 2           # batch size
D_MODEL = 4     # d_model
SEQ_LEN = 10    # seq_len
H = 2           # h

def test_input_embeddings() -> None:
    input_embedding = InputEmbeddings(D_MODEL, 20)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    y = input_embedding(x)

    assert y.size() == (2, 4, D_MODEL)

def test_positional_encoding() -> None:
    pos_encoding = PositionalEncoding(D_MODEL, SEQ_LEN, dropout=0.2)

    assert pos_encoding.pe.size() == (1, SEQ_LEN, D_MODEL)

    x = torch.zeros([B, SEQ_LEN, D_MODEL])    # (B, seq_len, d_model)
    y = pos_encoding(x)
    assert y.size() == (B, SEQ_LEN, D_MODEL)

def test_layer_normalization() -> None:
    layer_norm = LayerNormalization()
    x = torch.rand(B, SEQ_LEN, D_MODEL)
    y = layer_norm(x)

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def test_feedforward_block() -> None:
    ff = FeedForwardBlock(D_MODEL, SEQ_LEN, dropout=0.2)
    x = torch.rand(B, SEQ_LEN, D_MODEL)
    y = ff(x)

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def test_multi_head_attention_block() -> None:
    mha = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
    x = torch.rand(B, SEQ_LEN, D_MODEL)
    mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)
    y = mha(x, x, x, mask)

    assert y.size() == (B, SEQ_LEN, D_MODEL)
    assert mha.attention_scores.size() == (B, H, SEQ_LEN, SEQ_LEN)

def test_residual_connection() -> None:
    residual_connection = ResidualConnection(dropout=0.2)
    mha = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)

    x = torch.rand(B, SEQ_LEN, D_MODEL)
    mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    y = residual_connection(x, sublayer=lambda t: mha(t, t, t, mask))

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def test_encoder_block() -> None:
    mha = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
    ff = FeedForwardBlock(D_MODEL, SEQ_LEN, dropout=0.2)
    encoder_block = EncoderBlock(mha, ff, dropout=0.2)

    x = torch.rand(B, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    y = encoder_block(x, src_mask)

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def create_encoder() -> Encoder:
    layers = []
    for _ in range(6):
        mha = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
        ff = FeedForwardBlock(D_MODEL, SEQ_LEN, dropout=0.2)
        encoder_block = EncoderBlock(mha, ff, dropout=0.2)

        layers.append(encoder_block)

    encoder = Encoder(nn.ModuleList(layers))

    return encoder

def test_encoder() -> None:
    encoder = create_encoder()

    x = torch.rand(B, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    y = encoder(x, src_mask)

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def test_decoder_block() -> None:
    self_attention_block = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
    cross_attention_block = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
    ff = FeedForwardBlock(D_MODEL, SEQ_LEN, dropout=0.2)
    decoder_block = DecoderBlock(self_attention_block, cross_attention_block, ff, dropout=0.2)

    x = torch.rand(B, SEQ_LEN, D_MODEL)
    encoder_output = torch.rand(B, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)
    tgt_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    y = decoder_block(x, encoder_output, src_mask, tgt_mask)

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def create_decoder() -> Decoder:
    layers = []
    for _ in range(6):
        self_attention_block = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
        cross_attention_block = MultiHeadAttentionBlock(D_MODEL, H, dropout=0.2)
        ff = FeedForwardBlock(D_MODEL, SEQ_LEN, dropout=0.2)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, ff, dropout=0.2)

        layers.append(decoder_block)

    decoder = Decoder(layers)

    return decoder

def test_decoder() -> None:
    decoder = create_decoder()

    x = torch.rand(B, SEQ_LEN, D_MODEL)
    encoder_output = torch.rand(B, SEQ_LEN, D_MODEL)
    src_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)
    tgt_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    y = decoder(x, encoder_output, src_mask, tgt_mask)

    assert y.size() == (B, SEQ_LEN, D_MODEL)

def test_projection_layer() -> None:
    projection_layer = ProjectionLayer(D_MODEL, 100)
    x = torch.rand(B, SEQ_LEN, D_MODEL)
    y = projection_layer(x)

    assert y.size() == (B, SEQ_LEN, 100)

def test_transformer() -> None:
    src_vocab_size = 100
    tgt_vocab_size = 200
    
    encoder = create_encoder()
    decoder = create_decoder()
    src_embed = InputEmbeddings(D_MODEL, src_vocab_size)
    tgt_embed = InputEmbeddings(D_MODEL, tgt_vocab_size)
    src_pos = PositionalEncoding(D_MODEL, SEQ_LEN, dropout=0.2)
    tgt_pos = PositionalEncoding(D_MODEL, SEQ_LEN, dropout=0.2)
    projection_layer = ProjectionLayer(D_MODEL, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    src = torch.randint(low=0, high=src_vocab_size, size=(B, SEQ_LEN))
    src_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    encoder_output = transformer.encode(src, src_mask)
    assert encoder_output.size() == (B, SEQ_LEN, D_MODEL)

    tgt = torch.randint(low=0, high=tgt_vocab_size, size=(B, SEQ_LEN))
    tgt_mask = torch.ones(B, H, SEQ_LEN, SEQ_LEN)

    decoder_output = transformer.decode(encoder_output, src_mask, tgt, tgt_mask)
    assert decoder_output.size() == (B, SEQ_LEN, D_MODEL)

    result = transformer.project(decoder_output)
    assert result.size() == (B, SEQ_LEN, tgt_vocab_size)
