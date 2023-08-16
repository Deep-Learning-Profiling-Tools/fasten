import pytest
import torch
import triton

from fasten import compact_tensor_types


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compact_tensor_types(device: str):
    data = torch.tensor([[1, 2], [3, 4], [5, 6]], device=device)
    types = torch.tensor([2, 1, 2], dtype=torch.long, device=device)
    data_sorted, tensor_slice = compact_tensor_types(data, types, device=device)
    assert data_sorted[0].tolist() == [3, 4]
    type_slice = tensor_slice.get_slice_from_type(2)
    assert type_slice[1] == 2
    assert type_slice[2] == 1
    assert type_slice[3] == 3
    index_slice = tensor_slice.get_slice_from_index(1)
    assert torch.equal(index_slice, type_slice)
    type = tensor_slice.get_type_from_index(1)
    assert type == 2


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tile_size', [1, 2, 3, 16, 128])
def test_tiling(tile_size: int, device: str):
    data = torch.ones((128, 128), device=device)
    types = torch.zeros(128, dtype=torch.long, device=device)
    types[63:90] = 2
    types[90:128] = 3
    types[0:63] = 1
    _, tensor_slice = compact_tensor_types(data, types, device=device)
    tensor_tile = tensor_slice.tiling(tile_size)
    num_slices = triton.cdiv(90 - 63, tile_size) + triton.cdiv(128 - 90, tile_size) + triton.cdiv(63, tile_size)
    assert len(tensor_tile) == num_slices
