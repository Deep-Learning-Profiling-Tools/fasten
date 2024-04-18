import pytest
import torch
import triton

from fasten import compact_tensor_types


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
    avg_tile_size = 128 / num_slices
    # calculate stddev tile size
    stddev_tile_size = 0
    for i in range(len(tensor_tile)):
        start = tensor_tile.get_slice_from_index(i, is_tensor=False).start
        end = tensor_tile.get_slice_from_index(i, is_tensor=False).stop
        stddev_tile_size += ((end - start) - avg_tile_size) ** 2
    stddev_tile_size = (stddev_tile_size / num_slices) ** 0.5
    assert len(tensor_tile) == num_slices
    assert torch.equal(tensor_tile.avg_tile_size, avg_tile_size)
    assert torch.equal(tensor_tile.stddev_tile_size, stddev_tile_size)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tile_size', [1, 2, 3, 16, 128])
@pytest.mark.parametrize('block_size', [1])
def test_tile_slice_mapping(tile_size: int, block_size: int, device: str):
    data = torch.ones((128, 128), device=device)
    types = torch.zeros(128, dtype=torch.int, device=device)
    types[63:90] = 2
    types[90:128] = 3
    types[0:63] = 1
    tensor_slice = compact_tensor_types(data, types, device=device)
    tensor_tile = tensor_slice.tiling(tile_size, block_size=block_size)
    slice_tile_mapping = tensor_tile.slice_tile_mapping
    assert slice_tile_mapping[-1][2] == len(tensor_tile)
