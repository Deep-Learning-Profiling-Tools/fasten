import torch

from .utils import *


class FastenBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, input_slices: torch.Tensor,
                weight: torch.Tensor, weight_slices: torch.Tensor, engine_name=None):
        module = get_module(input.is_cuda)
        engine = get_engine(engine_name=engine_name)
        output = module.forward(
            input, input_slices, weight, weight_slices, engine)
        variables = [input, input_slices, weight, weight_slices, engine]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        module = get_module(grad.is_cuda)
        input, input_slices, weight, weight_slices, engine = ctx.saved_tensors
        grad_output, grad_weight = module.backward(
            grad, input, input_slices, weight, weight_slices, engine)
        return grad_output, grad_weight
