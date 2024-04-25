import torch

from model import InputEmbeddings, PositionalEncoding

D_MODEL = 4
SEQ_LEN = 10

def test_input_embeddings() -> None:
    input_embedding = InputEmbeddings(D_MODEL, 20)
    x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
    y = input_embedding(x)

    assert y.size() == (2, 4, D_MODEL)

def test_positional_encoding() -> None:
    pos_encoding = PositionalEncoding(D_MODEL, SEQ_LEN, dropout=0.2)

    assert pos_encoding.pe.size() == (1, SEQ_LEN, D_MODEL)

    x = torch.zeros([3, 8, D_MODEL])    # (B, seq_len, d_model)
    y = pos_encoding(x)
    assert y.size() == (3, 8, D_MODEL)
