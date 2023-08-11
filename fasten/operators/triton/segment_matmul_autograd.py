import torch 
import triton
import triton.language as tl
from segment_matmul import *
from matmul import *
from split_matmul import *
import pyg_lib

class SegmentMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, ptr, other):
        ctx.save_for_backward(inputs, ptr, other)
        triton_output= triton_segment_matmul(inputs, ptr, other)
        return triton_output

    @staticmethod
    def backward(ctx, grad_output):
        saved_variables= ctx.saved_tensors
        inputs= saved_variables[0]
        ptr= saved_variables[1]
        other= saved_variables[2]
        grad_out= grad_output

        other_t= other.transpose(-2,-1)

        input_grad= triton_segment_matmul(grad_out, ptr, other_t)
        # print(input_grad.shape)

        inputs_t = inputs.transpose(-2,-1)
        other_grad = split_matmul(inputs_t, grad_out, ptr)
        # print(input_grad.shape)
        # print(other_grad.shape)
        
        return input_grad, None, other_grad

segmentmatmul= SegmentMatmul.apply
torch.manual_seed(0)
inputs = torch.randn(148082, 16, requires_grad=True).cuda()
inputs.retain_grad()
ptr = torch.tensor([     0,  22484,  44968,  63602,  82236,  91553, 100870, 110059, 119248,
        128437, 137626, 141152, 144678, 145018, 145358, 145663, 145968, 146261,
        146554, 146837, 147120, 147323, 147526, 147546, 147566, 147627, 147688,
        147743, 147798, 147833, 147868, 147893, 147918, 147943, 147968, 147985,
        148002, 148005, 148008, 148021, 148034, 148047, 148060, 148066, 148072,
        148077, 148082]).cuda()
other = torch.randn(46,16,2, requires_grad=True).cuda()
other.retain_grad()
triton_output= segmentmatmul(inputs, ptr, other)
triton_output.mean().backward()
inputst = torch.randn(148082, 16, requires_grad=True).cuda()
inputst.retain_grad()
ptrt = torch.tensor([     0,  22484,  44968,  63602,  82236,  91553, 100870, 110059, 119248,
        128437, 137626, 141152, 144678, 145018, 145358, 145663, 145968, 146261,
        146554, 146837, 147120, 147323, 147526, 147546, 147566, 147627, 147688,
        147743, 147798, 147833, 147868, 147893, 147918, 147943, 147968, 147985,
        148002, 148005, 148008, 148021, 148034, 148047, 148060, 148066, 148072,
        148077, 148082]).cuda()
othert = torch.randn(46,16,2, requires_grad=True).cuda()
othert.retain_grad()
torch_output= pyg_lib.ops.segment_matmul(inputst, ptrt, othert)
torch_output.mean().backward()
print("Max Difference:",torch.max(torch.abs(inputs.grad-inputst.grad)))