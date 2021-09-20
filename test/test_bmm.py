from torch._C import dtype
import fasten
import torch

device = torch.device('cuda:0')
ops = fasten.HeteroOps(device)


def correctness():
    input_slice = [(1, slice(0, 2)), (2, slice(2, 3))]
    input = torch.tensor([[1, 2], [3, 4], [5, 6]],
                         device=device, dtype=torch.float)
    other_slice = [(1, slice(0, 1)), (2, slice(1, 2))]
    other = torch.tensor([[[7, 8], [1, 2]], [[3, 4], [5, 6]]],
                         device=device, dtype=torch.float)
    input_tensor_slice = fasten.TensorSlice(input, input_slice)
    other_tensor_slice = fasten.TensorSlice(other, other_slice)
    output = ops.bmm(input_tensor_slice, other_tensor_slice)
    truth = torch.tensor([[9, 12], [25, 32], [45, 56]],
                         device=device, dtype=torch.float)
    assert(torch.all(output == truth).item() is True)


# 1. Compare single stream vs multiple streams
# 2. Compare bmm + index vs heterogenous bmm
def speed():
    pass


def test_bmm():
    correctness()
