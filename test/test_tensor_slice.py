import pytest
import torch
import triton

from fasten import compact_tensor_types
from fasten.utils import TilingMethod


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dim', [0, 1])
def test_compact_tensor_types(device: str, dim: int):
    data = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]], device=device)
    types = torch.tensor([2, 1, 2], dtype=torch.long, device=device)
    tensor_slice = compact_tensor_types(data, types, dim=dim, device=device)
    if dim == 0:
        assert tensor_slice.data[0].tolist() == [3, 4, 5]
    else:
        assert tensor_slice.data[:, 0].tolist() == [2, 4, 6]
    slice = tensor_slice.get_slice_from_type(2)
    assert slice[0] == 1
    assert slice[1] == 3
    index_slice = tensor_slice.get_slice_from_index(1)
    assert torch.equal(index_slice, slice)
    type = tensor_slice.get_type_from_index(1)
    assert type == 2


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tile_size', [1, 2, 3, 16, 128])
def test_tiling_default(tile_size: int, device: str):
    data = torch.ones((128, 128), device=device)
    types = torch.zeros(128, dtype=torch.long, device=device)
    types[63:90] = 2
    types[90:128] = 3
    types[0:63] = 1
    tensor_slice = compact_tensor_types(data, types, device=device)
    tensor_tile = tensor_slice.tiling(tile_size)
    num_slices = triton.cdiv(90 - 63, tile_size) + triton.cdiv(128 - 90, tile_size) + triton.cdiv(63, tile_size)
    assert len(tensor_tile) == num_slices


@pytest.mark.parametrize('tile_size', [16])
def test_tiling_balanced(tile_size: int):
    data = torch.ones((512, 128))
    types = torch.zeros(512, dtype=torch.long)
    types[0:8] = 1
    types[8:64] = 2
    types[64:128] = 3
    types[128:256] = 4
    types[256:280] = 5
    types[280:512] = 6
    tensor_slice = compact_tensor_types(data, types)
    tensor_tile = tensor_slice.tiling(tile_size, method=TilingMethod.BALANCED)
    assert tensor_tile.num_blocks == 5
