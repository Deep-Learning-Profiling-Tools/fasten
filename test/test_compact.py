import fasten
import torch

device = torch.device('cpu')
ops = fasten.HeteroOps(device)


def compact_ascending():
    data = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.Tensor([2, 1, 1])
    tensor_slice = ops.compact(data, types)
    slice_truth = [(1, slice(0, 2)), (2, slice(2, 3))]
    data_truth = torch.Tensor([[3, 4], [5, 6], [1, 2]])
    assert(torch.all(tensor_slice.tensor == data_truth).item() is True)
    assert(tensor_slice.type_slices == slice_truth)


def compact_descending():
    data = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.Tensor([2, 1, 1])
    tensor_slice = ops.compact(data, types, True)
    slice_truth = [(2, slice(0, 1)), (1, slice(1, 3))]
    data_truth = torch.Tensor([[1, 2], [3, 4], [5, 6]])
    assert(torch.all(tensor_slice.tensor == data_truth).item() is True)
    assert(tensor_slice.type_slices == slice_truth)


def test_compact():
    compact_descending()
    compact_ascending()
