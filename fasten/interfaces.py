import torch

from .modules import *


class FastenBmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, input_slices: torch.tensor,
                other: torch.tensor, other_slices: torch.tensor, output: torch.tensor, engine_name=None):
        module = get_module(input.is_cuda)
        engine = get_engine(module, engine_name=engine_name)
        output = module.bmm_forward(
            input, input_slices, other, other_slices, output, engine)
        variables = [input, other]
        ctx.input_slices = input_slices
        ctx.other_slices = other_slices
        ctx.engine = engine
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.tensor):
        module = get_module(grad.is_cuda)
        input, other = ctx.saved_tensors
        grad_output, grad_other = module.bmm_backward(
            grad, input, ctx.input_slices, other, ctx.other_slices, ctx.engine)
        return grad_output, None, grad_other, None, None, None


class FastenBmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, input_slices: torch.tensor,
                other: torch.tensor, other_slices: torch.tensor, output: torch.tensor):
        module = get_module(input.is_cuda)
        output = module.bmul_forward(
            input, input_slices, other, other_slices, output)
        variables = [input, input_slices, other, other_slices]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.tensor):
        module = get_module(grad.is_cuda)
        input, input_slices, other, other_slices = ctx.saved_tensors
        grad_output, grad_other = module.bmul_backward(
            grad, input, input_slices, other, other_slices)
        return grad_output, grad_other


class FastenBdiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, input_slices: torch.tensor,
                other: torch.tensor, other_slices: torch.tensor, output: torch.tensor):
        module = get_module(input.is_cuda)
        output = module.bdiv_forward(
            input, input_slices, other, other_slices, output)
        variables = [input, input_slices, other, other_slices]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.tensor):
        module = get_module(grad.is_cuda)
        input, input_slices, other, other_slices = ctx.saved_tensors
        grad_output, grad_other = module.bdiv_backward(
            grad, input, input_slices, other, other_slices)
        return grad_output, grad_other


class FastenBadd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, input_slices: torch.tensor,
                other: torch.tensor, other_slices: torch.tensor, output: torch.tensor):
        module = get_module(input.is_cuda)
        output = module.badd_forward(
            input, input_slices, other, other_slices, output)
        variables = [input, input_slices, other, other_slices]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.tensor):
        module = get_module(grad.is_cuda)
        input, input_slices, other, other_slices = ctx.saved_tensors
        grad_output, grad_other = module.badd_backward(
            grad, input, input_slices, other, other_slices)
        return grad_output, grad_other


class FastenBsub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.tensor, input_slices: torch.tensor,
                other: torch.tensor, other_slices: torch.tensor, output: torch.tensor):
        module = get_module(input.is_cuda)
        output = module.bsub_forward(
            input, input_slices, other, other_slices, output)
        variables = [input, input_slices, other, other_slices]
        ctx.save_for_backward(*variables)

        return output

    @staticmethod
    def backward(ctx, grad: torch.tensor):
        module = get_module(grad.is_cuda)
        input, input_slices, other, other_slices = ctx.saved_tensors
        grad_output, grad_other = module.bsub_backward(
            grad, input, input_slices, other, other_slices)
        return grad_output, grad_other
