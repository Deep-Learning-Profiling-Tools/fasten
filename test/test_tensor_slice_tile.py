import torch
from torch.functional import Tensor
from fasten import Ops as ops
from fasten import TensorSliceTile


def test_iterator():
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    types = torch.tensor([3, 3, 2, 2, 5])
    _, tensor_slice = ops.compact(data, types)
    for i, tile in enumerate(TensorSliceTile(tensor_slice, step=2)):
        if i == 0:
            tile_truth = torch.tensor([[2, 0, 2], [3, 2, 4]])
            assert(torch.all(tile_truth == tile).item() is True)
        elif i == 1:
            tile_truth = torch.tensor([[5, 4, 5]])
            assert(torch.all(tile_truth == tile).item() is True)
