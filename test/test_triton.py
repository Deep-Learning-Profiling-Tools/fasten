import pytest
import torch

from fasten import compact_tensor_types
from fasten.operators import triton_ops
from fasten.utils import TilingMethod


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("M", [128, 1024])
@pytest.mark.parametrize("T", [16, 33])
@pytest.mark.parametrize("tile_size", [16, 64])
@pytest.mark.parametrize("block_size", [1, 4])
@pytest.mark.parametrize("K", [16, 80])
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("tiling_method", ["default"])
@pytest.mark.parametrize("trunc", [False])
def test_segment_matmul(M: int, K: int, T: int, phase: str, dtype: str, tile_size: int, block_size: int, device: str, tiling_method: str, trunc: bool) -> None:
    dtype = getattr(torch, dtype)
    data = torch.randn((M, K), dtype=dtype, device=device)
    types = torch.randint(0, T, (M,), device=device, dtype=torch.int)
    tensor_slice = compact_tensor_types(data, types, device=device)
    other = torch.randn((T, K, K), device=device, dtype=dtype)
    tiling_method = getattr(TilingMethod, tiling_method.upper())
    start_and_type = None
    end_and_next = None
    contiguous_flags = None
    if phase == "forward":
        input_tiles = tensor_slice.tiling(tile_size, method=tiling_method, block_size=block_size)
        if trunc:
            start_and_type, end_and_next, contiguous_flags = input_tiles.trunc()
        output = triton_ops.segment_matmul_forward(tensor_slice.data, other, input_tiles.slices, input_slices=tensor_slice.slices, tile_size=tile_size, out_dtype=torch.float32, num_blocks=input_tiles.num_blocks, block_size=input_tiles.block_size, start_and_type=start_and_type, end_and_next=end_and_next, contiguous_flags=contiguous_flags)
        output_ref = torch.zeros((M, K), dtype=dtype, device="cuda")
        for i in range(len(tensor_slice)):
            s = tensor_slice.get_slice_from_index(i, is_tensor=False)
            t = tensor_slice.get_type_from_index(i, is_tensor=False)
            output_ref[s] = torch.matmul(tensor_slice.data[s], other[t])
        torch.testing.assert_close(output, output_ref, atol=1e-1, rtol=1e-2)
    elif phase == "backward":
        input_tiles = tensor_slice.tiling(tile_size, method=tiling_method, block_size=block_size)
        if trunc:
            start_and_type, end_and_next, contiguous_flags = input_tiles.trunc()
        output = triton_ops.segment_matmul_forward(tensor_slice.data, other, input_tiles.slices, input_slices=tensor_slice.slices, tile_size=tile_size, num_blocks=input_tiles.num_blocks, block_size=input_tiles.block_size, start_and_type=start_and_type, end_and_next=end_and_next, contiguous_flags=contiguous_flags)
        output_grad = torch.randn_like(output)
        grad_input, grad_other = triton_ops.segment_matmul_backward(tensor_slice.data, output_grad, other, input_tiles.slices, input_slices=tensor_slice.slices, tile_size=tile_size, num_blocks=input_tiles.num_blocks, block_size=input_tiles.block_size, start_and_type=start_and_type, end_and_next=end_and_next, contiguous_flags=contiguous_flags)
        sorted_data_grad_ref = torch.zeros_like(data, dtype=dtype)
        other_grad_ref = torch.zeros_like(other, dtype=dtype)
        for i in range(len(tensor_slice)):
            s = tensor_slice.get_slice_from_index(i, is_tensor=False)
            t = tensor_slice.get_type_from_index(i, is_tensor=False)
            sorted_data_grad_ref[s] = torch.matmul(output_grad[s], other[t].t())
            other_grad_ref[t] = torch.matmul(tensor_slice.data[s].t(), output_grad[s])
        torch.testing.assert_close(grad_input, sorted_data_grad_ref, atol=1e-1, rtol=1e-2)
        torch.testing.assert_close(grad_other, other_grad_ref, atol=1e-1, rtol=1e-2)
