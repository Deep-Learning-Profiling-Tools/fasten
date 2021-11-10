import fasten
import torch

device = torch.device('cpu')
ops = fasten.HeteroOps(device)


def compact_ascending():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.tensor([2, 1, 1])
    tensor_slice = ops.compact(data, types)
    slice_truth = torch.tensor([[1, 0, 2], [2, 2, 3]])
    data_truth = torch.tensor([[3, 4], [5, 6], [1, 2]])
    assert(torch.all(tensor_slice.tensor == data_truth).item() is True)
    assert(torch.all(tensor_slice.slices == slice_truth).item() is True)


def compact_descending():
    data = torch.tensor([[1, 2], [3, 4], [5, 6]])
    types = torch.tensor([2, 1, 1])
    tensor_slice = ops.compact(data, types, descending=True)
    slice_truth = torch.tensor([[2, 0, 1], [1, 1, 3]])
    data_truth = torch.tensor([[1, 2], [3, 4], [5, 6]])
    assert(torch.all(tensor_slice.tensor == data_truth).item() is True)
    assert(torch.all(tensor_slice.slices == slice_truth).item() is True)


def test_compact():
    compact_descending()
    compact_ascending()
