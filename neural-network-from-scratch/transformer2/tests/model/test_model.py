import tempfile
from pathlib import Path

import torch
from torch import Tensor, nn
import numpy as np
from numpy.testing import assert_allclose

from model.model import AttentionBlock, FeedForwardBlock, ResidualConnection, DecoderLayer, ProjectionLayer, MultiHeadAttentionBlock, \
    create_rotary_position_encoding_tensor, swap_adjacent_last_dim, apply_rotary_position_encoding, MiniTransformer


def test_swap_adjacent_last_dim() -> None:
    x = Tensor([
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        [
            [0.3, 0.4, 0.5, 0.6],
            [0.5, 0.6, 0.7, 0.8],
        ]
    ])

    y = swap_adjacent_last_dim(x)

    expected = np.array([
        [
            [0.2000, 0.1000, 0.4000, 0.3000],
            [0.6000, 0.5000, 0.8000, 0.7000]
        ],
        [
            [0.4000, 0.3000, 0.6000, 0.5000],
            [0.6000, 0.5000, 0.8000, 0.7000]
        ]
    ])

    assert_allclose(y.detach().numpy(), expected, atol=1e-8)


def test_apply_rotary_position_encoding() -> None:
    x = torch.ones([2, 3, 4], requires_grad=True)

    *_, n_seq, d_model = x.shape

    y = apply_rotary_position_encoding(x, d_model, n_seq)

    # make it preserves gradients
    loss = y.sum() / 2
    loss.backward()
    assert torch.all(x.grad != 0)   # type: ignore[arg-type]

    expected = np.array([
        [
            [ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.3818, -0.3012,  1.0099,  0.9900],
            [ 0.4932, -1.3254,  1.0198,  0.9798]
        ],
        [
            [ 1.0000,  1.0000,  1.0000,  1.0000],
            [ 1.3818, -0.3012,  1.0099,  0.9900],
            [ 0.4932, -1.3254,  1.0198,  0.9798]
        ]
    ])
    assert_allclose(y.detach(), expected, atol=1e-4)



def test_attention_block() -> None:
    weights = {
        "W_q": Tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ]),
        "W_k": Tensor([
            [0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7],
            [0.6, 0.9, 1.0]
        ]),
        "W_v": Tensor([
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1]
        ])
    }
    attn = AttentionBlock(d_model=3, weights=weights)
    x = Tensor([
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
        ]
    ]).float()
    assert x.shape == torch.Size([1, 2, 3])

    output = attn(x, x, x)
    # print(output)

    assert output.shape == torch.Size([1, 2, 3])
    assert_allclose(output.detach().numpy(), np.array([
        [
            [0.323377, 0.55091, 0.778443],
            [0.32497, 0.553697, 0.782425]
        ]
    ]), atol=1e-4)


def test_multi_head_attention_block() -> None:
    weights = {
        "W_q": Tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6]
        ]),
        "W_k": Tensor([
            [0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6, 1.7]
        ]),
        "W_v": Tensor([
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4],
            [1.5, 1.6, 1.7, 1.8]
        ]),
        "linear": Tensor([
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 1.1],
            [1.2, 1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8, 1.9]
        ]),
        "linear_bias": Tensor([0.0, 0.0, 0.0, 0.0])
    }

    x = Tensor([
        [
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9]
        ]
    ]).float()

    d_model = x.shape[-1]
    n_seq = x.shape[-2]
    h = d_model // 2

    # TODO rope_tensor is not used
    rope_tensor = create_rotary_position_encoding_tensor(d_model, n_seq)

    attn = MultiHeadAttentionBlock(d_model, n_seq, h, weights=weights)
    output = attn(x, x, x, mask=None)

    assert output.shape == torch.Size([1, n_seq, d_model])
    assert_allclose(output.detach().numpy(), np.array([
        [
            [ 5.711371,  9.487418, 13.263464, 17.03951 ],
            [ 5.902481,  9.828537, 13.754593, 17.680649],
            [ 6.10944 , 10.203999, 14.298557, 18.393116]
        ]
    ]), atol=1e-5)


def test_multi_head_attention_block_with_mask() -> None:
    weights = {
        "W_q": Tensor([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2],
            [1.3, 1.4, 1.5, 1.6]
        ]),
        "W_k": Tensor([
            [0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2, 1.3],
            [1.4, 1.5, 1.6, 1.7]
        ]),
        "W_v": Tensor([
            [0.3, 0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9, 1.0],
            [1.1, 1.2, 1.3, 1.4],
            [1.5, 1.6, 1.7, 1.8]
        ]),
        "linear": Tensor([
            [0.4, 0.5, 0.6, 0.7],
            [0.8, 0.9, 1.0, 1.1],
            [1.2, 1.3, 1.4, 1.5],
            [1.6, 1.7, 1.8, 1.9]
        ]),
        "linear_bias": Tensor([0.0, 0.0, 0.0, 0.0])
    }

    x = Tensor([
        [
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9]
        ]
    ]).float()

    mask = Tensor([
        [True, True, False]
    ]).bool()

    batch_size, n_seq, d_model = x.shape
    assert mask.shape[0] == batch_size
    assert mask.shape[1] == n_seq

    h = d_model // 2

    # TODO rope_tensor is not used
    rope_tensor = create_rotary_position_encoding_tensor(d_model, n_seq)

    attn = MultiHeadAttentionBlock(d_model, n_seq, h, weights=weights)
    output = attn(x, x, x, mask)

    assert output.shape == torch.Size([1, n_seq, d_model])
    assert_allclose(output.detach().numpy(), np.array([
        [
            [4.236, 7.084, 9.931999, 12.780001],
            [5.086045, 8.495882, 11.90572, 15.315557],
            [5.149142, 8.606613, 12.064085, 15.521557]
        ]
    ]), atol=1e-5)


def test_feed_forward_block() -> None:
    feed_forward = FeedForwardBlock(3, 6, dropout=0.1)
    x = Tensor([
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
        ]
    ]).float()
    output = feed_forward(x)

    assert output.shape == torch.Size([1, 2, 3])


def test_residual_connection() -> None:
    d_model = 3
    n_seq = 2
    dense = nn.Linear(d_model, d_model)
    resid_conn = ResidualConnection(d_model, n_seq, dropout=0.1)

    x = Tensor([
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
        ],
        [
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    ]).float()

    output = resid_conn(x, dense)

    assert output.shape == torch.Size([2, n_seq, d_model])
    for i in range(output.shape[0]):
        y = output[i].detach().numpy()
        assert abs(y.sum()) < 1e-6
        assert abs(y.std() - 1) < 1e-3


def test_decoder_layer() -> None:
    x = Tensor([
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        [
            [0.2, 0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8, 0.9],
        ]
    ]).float()

    batch_size, n_seq, d_model = x.shape
    h = 2

    decoder_layer = DecoderLayer(d_model, n_seq, h, dropout=0.1)

    output = decoder_layer(x)

    print(output)

    assert output.shape == torch.Size([batch_size, n_seq, d_model])


def test_projection_layer() -> None:
    d_model = 3
    n_seq = 2
    vocab_size = 4

    proj_layer = ProjectionLayer(d_model, vocab_size)

    x = Tensor([
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
        ],
        [
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    ]).float()

    output = proj_layer(x)
    print(output)

    assert output.shape == torch.Size([2, n_seq, vocab_size])
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            a = output[i, j].detach().numpy()
            assert abs(a.sum() - 1.0) < 1e-6


def test_mini_transformer() -> None:
    vocab_size = 100
    d_model = 4
    n_seq = 100
    batch_size = 3

    input = torch.randint(0, vocab_size, (batch_size, n_seq))

    transformer = MiniTransformer(vocab_size=vocab_size, d_model=d_model, n_seq=n_seq, h=2, n_layers=6, dropout=0.1)
    output = transformer(input)

    assert output.shape == torch.Size([batch_size, n_seq, vocab_size])


def state_dicts_allclose(sd1, sd2, rtol=1e-05, atol=1e-08):
    if sd1.keys() != sd2.keys():
        return False
    return all(torch.allclose(sd1[k], sd2[k], rtol=rtol, atol=atol) for k in sd1)


def test_save_load_model() -> None:
    model = MiniTransformer(vocab_size=10, d_model=4, n_seq=5, h=2, n_layers=6, dropout=0.1)
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        print("Path to temp file:", temp_file.name)
        temp_file_path = Path(temp_file.name)
        model.save(temp_file_path)

        loaded_model = MiniTransformer.load(temp_file_path)

        assert loaded_model.vocab_size == model.vocab_size
        assert loaded_model.d_model == model.d_model
        assert loaded_model.n_seq == model.n_seq
        assert loaded_model.h == model.h
        assert loaded_model.n_layers == model.n_layers
        assert loaded_model.dropout == model.dropout

        assert state_dicts_allclose(loaded_model.state_dict(), model.state_dict())
