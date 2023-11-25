import torch


def segment_matmul_forward(input: torch.Tensor, other: torch.Tensor, input_slices: torch.Tensor,
                           output: torch.Tensor = None):
    assert input.device == other.device, 'input, other and output must be on the same device'
    input_slices = input_slices.to('cpu')
    if output is None:
        output = torch.empty(input.shape[0], other.shape[2], device=input.device, dtype=input.dtype)
    for i in range(input_slices.shape[0]):
        t = input_slices[i, 1]
        a = input[input_slices[i, 2]:input_slices[i, 3]]
        b = other[t]
        c = output[input_slices[i, 2]:input_slices[i, 3]]
        torch.matmul(a, b, out=c)
    return output


def segment_matmul_backward_input(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor,
                                  input_slices: torch.Tensor, grad_input: torch.Tensor = None):
    assert input.device == other.device, 'input, other and output must be on the same device'
    input_slices = input_slices.to('cpu')
    grad_output = grad_output.contiguous()
    if grad_input is None:
        grad_input = torch.empty_like(input)
    for i in range(input_slices.shape[0]):
        t = input_slices[i, 1]
        a = grad_output[input_slices[i, 2]:input_slices[i, 3]]
        b = other[t]
        c = grad_input[input_slices[i, 2]:input_slices[i, 3]]
        torch.matmul(a, b.t(), out=c)
    return grad_input


def segment_matmul_backward_other(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor,
                                  input_slices: torch.Tensor, grad_other: torch.Tensor = None):
    assert input.device == other.device, 'input, other and output must be on the same device'
    input_slices = input_slices.to('cpu')
    grad_output = grad_output.contiguous()
    if grad_other is None:
        # grad_other might be sparse
        grad_other = torch.zeros_like(other)
    for i in range(input_slices.shape[0]):
        t = input_slices[i, 1]
        a = input[input_slices[i, 2]:input_slices[i, 3]]
        b = grad_output[input_slices[i, 2]:input_slices[i, 3]]
        c = grad_other[t]
        torch.matmul(a.t(), b, out=c)
    return grad_other
