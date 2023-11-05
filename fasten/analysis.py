import torch

from .tensor_slice import TensorSlice


def get_average_slice_len(tensor_slice: TensorSlice) -> float:
    """Get the average length of tensor slices."""
    slices = tensor_slice.slices
    return torch.mean(slices[:, 3] - slices[:, 2]).item()
