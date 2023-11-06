import torch


def get_average_slice_len(slices: torch.Tensor) -> float:
    """Get the average length of tensor slices."""
    return torch.mean(slices[:, 3] - slices[:, 2]).item()


def get_num_contiguous_slices(slices: torch.Tensor) -> int:
    """Get the number of contiguous slices."""
    return torch.sum(slices[:, 4] == 0).item()


def get_num_non_contiguous_slices(slices: torch.Tensor) -> int:
    """Get the number of contiguous slices."""
    return torch.sum(slices[:, 4] != 0).item()
