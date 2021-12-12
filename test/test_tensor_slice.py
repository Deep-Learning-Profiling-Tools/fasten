import torch
from fasten import Ops as ops

device = torch.device('cpu')


def test_get_slice():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.tensor([2, 1, 2])
    _, tensor_slice = ops.compact(data, types)
    sub_tensor_slice = tensor_slice.get_slice(1)
    assert(sub_tensor_slice == slice(0, 1))


def test_subslices():
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    types = torch.tensor([2, 1, 2, 3])
    _, tensor_slice = ops.compact(data, types)
    sub_tensor_slice = tensor_slice.subslices(slice(1, 3))
    assert(tensor_slice[0, 0] == 1)
    assert(tensor_slice.get_slice(2) == slice(1, 3))
    assert(sub_tensor_slice.get_slice(2) == slice(1, 3))


def test_extract():
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    types = torch.tensor([2, 1, 2, 3])
    _, tensor_slice = ops.compact(data, types)
    sub_tensor_slice = tensor_slice.extract([3])
    assert(sub_tensor_slice[0, 0] == 3)
    assert(sub_tensor_slice[0, 1] == 0)
    assert(sub_tensor_slice[0, 2] == 1)
    tensor_slice[2, 0] = 1
    assert(sub_tensor_slice[0, 0] == 3)


def test_expand():
    data = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    types = torch.tensor([2, 1, 2, 3])
    _, tensor_slice = ops.compact(data, types)
    tensor = tensor_slice.expand()
    truth = torch.tensor([1, 2, 2, 3])
    assert(torch.all(tensor == truth).item() is True)
