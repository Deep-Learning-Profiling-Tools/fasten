import torch
from fasten import Ops as ops

device = torch.device('cpu')


def test_compact_descending():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.tensor([2, 1, 1])
    tensor, tensor_slice = ops.compact(data, types, descending=True)
    slice_truth = torch.tensor([[2, 0, 1], [1, 1, 3]])
    data_truth = torch.tensor([[1, 2], [3, 4], [5, 6]])
    assert(torch.all(tensor == data_truth).item() is True)
    assert(torch.all(tensor_slice == slice_truth).item() is True)


def test_compact_ascending():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.tensor([2, 1, 1])
    tensor, tensor_slice = ops.compact(data, types)
    slice_truth = torch.tensor([[1, 0, 2], [2, 2, 3]])
    data_truth = torch.tensor([[3, 4], [5, 6], [1, 2]])
    assert(torch.all(tensor == data_truth).item() is True)
    assert(torch.all(tensor_slice == slice_truth).item() is True)


def test_compact_type_dim():
    data = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    types = torch.tensor([2, 1, 1, 1])
    tensor, tensor_slice = ops.compact(data, types, type_dim=1)
    slice_truth = torch.tensor([[1, 0, 3], [2, 3, 4]])
    data_truth = torch.tensor([[2, 3, 4, 1], [6, 7, 8, 5]])
    assert(torch.all(tensor == data_truth).item() is True)
    assert(torch.all(tensor_slice == slice_truth).item() is True)
