import pytest
import torch

from fasten import compact_tensor_types
from fasten.stats import get_matmul_flops


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_matmul_flops(device):
    data = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7, 8]], device=device)
    types = torch.tensor([2, 1, 2], dtype=torch.int, device=device)
    tensor_slice = compact_tensor_types(data, types, device=device)
    weight = torch.randn((3, 4, 5), device=device)
    flops = get_matmul_flops(tensor_slice, weight)
    flops_ref = 2 * 4 * 5 * 2 + 4 * 5 * 2
    assert flops == flops_ref, f"{flops} != {flops_ref}"
