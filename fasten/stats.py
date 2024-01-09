import torch

from .tensor_slice import TensorSlice


def get_matmul_flops(input: TensorSlice, weight: torch.Tensor):
    assert weight.dim() == 3, f"weight dim should be 3, got {weight.dim()}"
    flops = 0
    for i in range(len(input)):
        s = input.get_slice_from_index(i, is_tensor=False)
        if s.stop - s.start == 0:
            continue
        flops += (s.stop - s.start) * weight.shape[1] * weight.shape[2] * 2
    return flops


def get_matmul_bytes(input: TensorSlice, weight: torch.Tensor):
    assert weight.dim() == 3, f"weight dim should be 3, got {weight.dim()}"
    bytes = 0
    for i in range(len(input)):
        s = input.get_slice_from_index(i, is_tensor=False)
        if s.stop - s.start == 0:
            continue
        # input
        bytes += (s.stop - s.start) * weight.shape[1] * weight.element_size()
        # weight
        bytes += weight.shape[1] * weight.shape[2] * weight.element_size()
        # output
        bytes += (s.stop - s.start) * weight.shape[2] * weight.element_size()
    return bytes
