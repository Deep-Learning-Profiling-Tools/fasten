import pytest
import torch

from fasten import compact_tensor_types
from fasten.operators import triton_ops

slices0 = [slice(0, 63), slice(63, 90), slice(90, 128)]
slices1 = [slice(0, 127), slice(127, 256), slice(256, 257), slice(257, 512)]


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("slices", [slices0, slices1])
@pytest.mark.parametrize("tile_size", [16, 32, 64])
@pytest.mark.parametrize("hidden_size", [16, 32, 64, 96])
def test_segment_matmul(slices: list, phase: str, dtype: str, tile_size: int, hidden_size: int) -> None:
    dtype = getattr(torch, dtype)
    M = sum([s.stop - s.start for s in slices])
    data = torch.randn((M, hidden_size), dtype=dtype, device="cuda")
    types = torch.zeros(M, dtype=torch.long, requires_grad=False, device="cuda")
    for i, s in enumerate(slices):
        types[s] = i
    sorted_data, tensor_slice = compact_tensor_types(data, types, device="cuda")
    other = torch.randn((len(slices), hidden_size, hidden_size), dtype=dtype, device="cuda")
    if phase == "forward":
        input_tiles = tensor_slice.tiling(tile_size)
        output = triton_ops.segment_matmul_forward(sorted_data, tensor_slice.slices, other, input_tiles.slices, tile_size=tile_size)
        output_ref = torch.zeros((M, hidden_size), dtype=dtype, device="cuda")
        for i in range(len(slices)):
            s = slices[i]
            output_ref[s] = torch.matmul(data[s], other[i])
        assert torch.allclose(output, output_ref, atol=1e-1, rtol=1e-1)
    elif phase == "backward":
        input_tiles = tensor_slice.tiling(tile_size)
        output = triton_ops.segment_matmul_forward(sorted_data, tensor_slice.slices, other, input_tiles.slices, tile_size=tile_size)
        output_grad = torch.randn_like(output)
        grad_input, grad_other = triton_ops.segment_matmul_backward(sorted_data, tensor_slice.slices, output_grad, other, input_tiles.slices, tile_size=tile_size)
        sorted_data_grad_ref = torch.zeros_like(data, dtype=dtype)
        other_grad_ref = torch.zeros_like(other, dtype=dtype)
        for i in range(len(slices)):
            s = slices[i]
            sorted_data_grad_ref[s] = torch.matmul(output_grad[s], other[i].t())
            other_grad_ref[i] = torch.matmul(sorted_data[s].t(), output_grad[s])
        assert torch.allclose(grad_input, sorted_data_grad_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(grad_other, other_grad_ref, atol=1e-1, rtol=1e-1)
