import pytest
import torch
import csv

from fasten import Engine, compact_tensor_types, ops

def read_slices_from_csv(csv_file):
    slices = []

    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start = int(row["Start"])
            end = int(row["End"])
            slices.append(slice(start, end))

    return slices

slices0 = [slice(0, 63), slice(63, 90), slice(90, 128)]
slices1 = [slice(0, 127), slice(127, 256), slice(256, 257), slice(257, 512)]

AIFB = read_slices_from_csv('AIFB.csv')
AM = read_slices_from_csv('AM.csv')
BGS = read_slices_from_csv('BGS.csv')
DBLP = read_slices_from_csv('DBLP.csv')
MUTAG = read_slices_from_csv('MUTAG.csv')


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("engine", [Engine.TORCH, Engine.TRITON])
@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])   
@pytest.mark.parametrize("slices", [slices0, slices1, AIFB, AM, BGS, DBLP, MUTAG])
@pytest.mark.parametrize("K", [16, 32, 48, 64, 80])
def test_segment_matmul(K: int, slices: list, engine: Engine, device: str, phase: str, dtype: str) -> None:
    if engine == Engine.TRITON and device == "cpu":
        pytest.skip("Triton does not support CPU inference")
    if device == "cpu" and dtype == "float16":
        pytest.skip("CPU does not support FP16")
    dtype = getattr(torch, dtype)
    M = sum([s.stop - s.start for s in slices])
    data = torch.randn((M, K), device=device, dtype=dtype)
    types = torch.zeros(M, dtype=torch.long, device=device, requires_grad=False)
    for i, s in enumerate(slices):
        types[s] = i
    sorted_data, tensor_slice = compact_tensor_types(data, types, device=device)
    other = torch.randn((len(slices), K, K), device=device, dtype=dtype)
    if phase == "forward":
        output = ops.fasten_segment_matmul(sorted_data, tensor_slice.slices, other, engine, tensor_slice)
        output_ref = torch.zeros((M, K), device=device, dtype=dtype)
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
