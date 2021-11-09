import torch

import bmm_cpp
import bmm_cuda


class FastenBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, input_slices: torch.Tensor, weight: torch.Tensor, weight_slices: torch.Tensor):
        if input.is_cuda:
            output = bmm_cuda.forward(
                input, input_slices, weight, weight_slices)
        else:
            output = bmm_cpp.forward(
                input, input_slices, weight, weight_slices)
        variables = [input, input_slices, weight, weight_slices]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if grad.is_cuda:
            grad_output, grad_weight = bmm_cuda.backward(
                grad, *ctx.savedtensors)
        else:
            grad_output, grad_weight = bmm_cpp.backward(
                grad, *ctx.savedtensors)
        return grad_output, grad_weight
