import pytest
import torch

from fasten import Engine, compact_tensor_types, ops


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("engine", [Engine.TORCH, Engine.TRITON])
@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_segment_matmul(engine: Engine, device: str, phase: str, dtype: str) -> None:
    if engine == Engine.TRITON and device == "cpu":
        pytest.skip("Triton does not support CPU inference")
    if device == "cpu" and dtype == "float16":
        pytest.skip("CPU does not support FP16")
    dtype = getattr(torch, dtype)
    data = torch.randn((128, 96), device=device, dtype=dtype)
    types = torch.zeros(128, dtype=torch.long, device=device, requires_grad=False)
    types[0:63] = 1
    types[63:90] = 2
    types[90:128] = 3
    sorted_data, tensor_slice = compact_tensor_types(data, types, device=device)
    other = torch.randn((3, 96, 64), device=device, dtype=dtype)
    if phase == "forward":
        output = ops.fasten_segment_matmul(sorted_data, tensor_slice.slices, other, engine, tensor_slice)
        output_ref = torch.zeros((128, 64), device=device, dtype=dtype)
        output_ref[0:63] = torch.matmul(data[0:63], other[0])
        output_ref[63:90] = torch.matmul(data[63:90], other[1])
        output_ref[90:128] = torch.matmul(data[90:128], other[2])
        assert torch.allclose(output, output_ref, atol=1e-1, rtol=1e-1)
    elif phase == "backward":
        sorted_data.requires_grad = True
        other.requires_grad = True
        output = ops.fasten_segment_matmul(sorted_data, tensor_slice.slices, other, engine, tensor_slice)
        output_grad = torch.randn_like(output)
        output.backward(output_grad)
        sorted_data_grad_ref = torch.zeros_like(data, dtype=dtype)
        other_grad_ref = torch.zeros_like(other, dtype=dtype)
        sorted_data_grad_ref[0:63] = torch.matmul(output_grad[0:63], other[0].t())
        sorted_data_grad_ref[63:90] = torch.matmul(output_grad[63:90], other[1].t())
        sorted_data_grad_ref[90:128] = torch.matmul(output_grad[90:128], other[2].t())
        other_grad_ref[0] = torch.matmul(sorted_data[0:63].t(), output_grad[0:63])
        other_grad_ref[1] = torch.matmul(sorted_data[63:90].t(), output_grad[63:90])
        other_grad_ref[2] = torch.matmul(sorted_data[90:128].t(), output_grad[90:128])
        assert torch.allclose(sorted_data.grad, sorted_data_grad_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(other.grad, other_grad_ref, atol=1e-1, rtol=1e-1)
