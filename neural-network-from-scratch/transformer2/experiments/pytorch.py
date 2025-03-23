import torch
from torch import Tensor

def matmul() -> None:
    t1 = Tensor([[
        [1,2,3],
        [4,5,6]
    ],[
        [2, 3, 4],
        [5, 6, 7]
    ]])
    t2 = Tensor([[
        [1,4],
        [2,5],
        [3,6]
    ], [
        [2, 5],
        [3, 6],
        [4, 7]
    ]])
    p = torch.matmul(t1, t2)
    print(p)


matmul()
