import torch

from fasten import compact_tensor_types
from fasten.stats import get_matmul_flops


def test_matmul_flops():
    data = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]])
    types = torch.tensor([2, 1, 2], dtype=torch.int)
    tensor_slice = compact_tensor_types(data, types)
    print(tensor_slice.slices)
    weight = torch.randn((3, 4, 5))
    print(get_matmul_flops(tensor_slice, weight))
