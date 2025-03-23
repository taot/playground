import math

import torch
from torch import Tensor
import numpy as np

def create_rotation_matrix(d: int, m: int) -> Tensor:
    assert d % 2 == 0
    R = np.zeros((d, d), dtype=np.float32)
    for i in range(d // 2):
        theta = math.pow(10000, -2 * i / d)
        R[2 * i, 2 * i] = math.cos(m * theta)
        R[2 * i, 2 * i + 1] = -1 * math.sin(m * theta)
        R[2 * i + 1, 2 * i] = math.sin(m * theta)
        R[2 * i + 1, 2 * i + 1] = math.cos(m * theta)

    t = torch.from_numpy(R)
    t.requires_grad_(False)

    return t


def create_rotation_matrix2(d: int, n_seq: int) -> Tensor:
    assert d % 2 == 0
    R = np.zeros((n_seq, d, d), dtype=np.float32)
    for j in range(n_seq):
        for i in range(d // 2):
            theta = math.pow(10000, -2 * i / d)
            R[j, 2 * i, 2 * i] = math.cos(j * theta)
            R[j, 2 * i, 2 * i + 1] = -1 * math.sin(j * theta)
            R[j, 2 * i + 1, 2 * i] = math.sin(j * theta)
            R[j, 2 * i + 1, 2 * i + 1] = math.cos(j * theta)

    t = torch.from_numpy(R)
    t.requires_grad_(False)

    return t


def apply_rotary_position(x: Tensor) -> Tensor:
    d = x.shape[-1]
    n_seq = x.shape[-2]
    # t1 = torch.tensor([
    #     [1, 2, 3, 4, 5, 6],
    #     [7, 8, 9, 10, 11, 12]
    # ])

    x2 = swap_adjacent_last_dim(x)

    t_cos = torch.zeros([n_seq, d], dtype=torch.float32)
    t_sin = torch.zeros([n_seq, d], dtype=torch.float32)
    for j in range(n_seq):
        for i in range(d // 2):
            theta = math.pow(10000, -2 * i / d)
            t_cos[j, 2 * i] = math.cos(theta * j)
            t_cos[j, 2 * i + 1] = math.cos(theta * j)
            t_sin[j, 2 * i] = math.sin(theta * j)
            t_sin[j, 2 * i + 1] = math.sin(theta * j) * -1

    output = torch.mul(x, t_cos) + torch.mul(x2, t_sin)

    # print(output)
    return output


def swap_adjacent_last_dim(tensor):
    # Get shape information
    last_dim = tensor.shape[-1]

    # Create indices for swapping in the last dimension
    indices = torch.arange(last_dim, device=tensor.device)
    indices[0::2], indices[1::2] = indices[1::2].clone(), indices[0::2].clone()

    # Handle odd length case
    # if last_dim % 2 == 1:
    #     indices[-1] = last_dim - 1

    # Use gather on the last dimension
    # This automatically handles the broadcasting across batch dimensions
    return torch.gather(tensor, -1, indices.expand(tensor.shape))


def element_wise_mul():
    t1 = Tensor([
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ]
    ])
    t2 = Tensor([
        [7, 8, 9],
    ])
    t3 = torch.mul(t1, t2)
    print(t3)


torch.set_printoptions(
    precision=4,       # Number of decimal places to display
    # threshold=1000000000,    # Total number of elements before summarizing
    # edgeitems=3,       # Number of items at the edges when summarizing
    linewidth=3000      # Characters per line
)

# t = create_rotation_matrix2(d=512, n_seq=1000)
# print(t)

# apply_rotary_position()

# x = torch.arange(24).reshape(2, 3, 4)
# print("Original:\n", x)
#
# # Swap elements in the last dimension
# swapped = swap_adjacent_last_dim(x)
# print("Swapped:\n", swapped)

# element_wise_mul()


# t1 = torch.tensor([
#     [1, 2, 3, 4, 5, 6],
#     [7, 8, 9, 10, 11, 12],
#     [13, 14, 15, 16, 17, 18],
# ])

t1 = torch.ones(torch.Size([2, 10, 8]))

t2 = apply_rotary_position(t1)
print(t2)
