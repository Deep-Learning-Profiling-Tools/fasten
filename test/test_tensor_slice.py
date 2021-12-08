import torch
from fasten import Ops as ops

device = torch.device('cpu')


def test_get():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.tensor([2, 1, 2])
    _, tensor_slice = ops.compact(data, types)
    sub_tensor_slice = tensor_slice.get(1)
    assert(sub_tensor_slice == slice(0, 1))


def test_subslices():
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    types = torch.tensor([2, 1, 2, 3])
    _, tensor_slice = ops.compact(data, types)
    sub_tensor_slice = tensor_slice.subslices(slice(1, 3))
    assert(tensor_slice[0, 0] == 1)
    assert(tensor_slice.get(2) == slice(1, 3))
    assert(sub_tensor_slice.get(2) == slice(1, 3))
