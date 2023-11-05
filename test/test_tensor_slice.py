import pytest
import torch
import triton
from utils import count_bits

from fasten import compact_tensor_types
from fasten.analysis import get_num_contiguous_slices


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dim', [0, 1])
def test_compact_tensor_types(device: str, dim: int):
    data = torch.tensor([[1, 2, 3], [3, 4, 5], [5, 6, 7]], device=device)
    types = torch.tensor([2, 1, 2], dtype=torch.int, device=device)
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
@pytest.mark.parametrize('block_size', [1])
def test_tiling_default(tile_size: int, block_size: int, device: str):
    data = torch.ones((128, 128), device=device)
    types = torch.zeros(128, dtype=torch.int, device=device)
    types[63:90] = 2
    types[90:128] = 3
    types[0:63] = 1
    tensor_slice = compact_tensor_types(data, types, device=device)
    tensor_tile = tensor_slice.tiling(tile_size, block_size=block_size)
    num_slices = triton.cdiv(90 - 63, tile_size) + triton.cdiv(128 - 90, tile_size) + triton.cdiv(63, tile_size)
    assert len(tensor_tile) == num_slices


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tile_size', [3, 16])
@pytest.mark.parametrize('block_size', [1, 4])
def test_trunc(device: str, tile_size: int, block_size: int):
    data = torch.ones((128, 128), device=device)
    types = torch.zeros(128, dtype=torch.int, device=device)
    types[0:64] = 1
    types[64:90] = 2
    types[90:128] = 3
    tensor_slice = compact_tensor_types(data, types, device=device)
    tensor_tile = tensor_slice.tiling(tile_size, block_size=block_size)
    start_and_type, end_and_next, contiguous_flags = tensor_tile.trunc()
    # count number of bits in contiguous_flags
    num_contiguous_slices = get_num_contiguous_slices(tensor_tile.slices)
    assert num_contiguous_slices == count_bits(contiguous_flags)
    assert torch.all(start_and_type >> 32 == tensor_tile.slices[:, 2])
    assert torch.all(start_and_type & 0xffffffff == tensor_tile.slices[:, 1])
    assert torch.all(end_and_next >> 32 == tensor_tile.slices[:, 3])
    assert torch.all(end_and_next & 0xffffffff == tensor_tile.slices[:, 4])
