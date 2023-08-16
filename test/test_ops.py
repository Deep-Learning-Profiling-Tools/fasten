import pytest
import torch

from fasten import Engine, compact_tensor_types, ops

slice0 = [slice(0, 63), slice(63, 90), slice(90, 128)]
slice1 = [slice(0, 127), slice(127, 256), slice(256, 257), slice(257, 512)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("engine", [Engine.TORCH, Engine.TRITON])
@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("slices", [slice0, slice1])
def test_segment_matmul(slices: list, engine: Engine, device: str, phase: str, dtype: str) -> None:
    if engine == Engine.TRITON and device == "cpu":
        pytest.skip("Triton does not support CPU inference")
    if device == "cpu" and dtype == "float16":
        pytest.skip("CPU does not support FP16")
    dtype = getattr(torch, dtype)
    M = sum([s.stop - s.start for s in slices])
    data = torch.randn((M, 96), device=device, dtype=dtype)
    types = torch.zeros(M, dtype=torch.long, device=device, requires_grad=False)
    for i, s in enumerate(slices):
        types[s] = i
    sorted_data, tensor_slice = compact_tensor_types(data, types, device=device)
    other = torch.randn((len(slices), 96, 64), device=device, dtype=dtype)
    if phase == "forward":
        output = ops.fasten_segment_matmul(sorted_data, tensor_slice.slices, other, engine, tensor_slice)
        output_ref = torch.zeros((M, 64), device=device, dtype=dtype)
        for i in range(len(slices)):
            s = slices[i]
            output_ref[s] = torch.matmul(data[s], other[i])
        assert torch.allclose(output, output_ref, atol=1e-1, rtol=1e-1)
    elif phase == "backward":
        sorted_data.requires_grad = True
        other.requires_grad = True
        output = ops.fasten_segment_matmul(sorted_data, tensor_slice.slices, other, engine, tensor_slice)
        output_grad = torch.randn_like(output)
        output.backward(output_grad)
        sorted_data_grad_ref = torch.zeros_like(data, dtype=dtype)
        other_grad_ref = torch.zeros_like(other, dtype=dtype)
        for i in range(len(slices)):
            s = slices[i]
            sorted_data_grad_ref[s] = torch.matmul(output_grad[s], other[i].t())
            other_grad_ref[i] = torch.matmul(sorted_data[s].t(), output_grad[s])
        assert torch.allclose(sorted_data.grad, sorted_data_grad_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(other.grad, other_grad_ref, atol=1e-1, rtol=1e-1)
