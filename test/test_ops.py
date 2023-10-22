import pyg_lib
import pytest
import torch
import triton
import json
import csv
from typing import Callable
from utils import read_slices_from_csv

from fasten import Engine, compact_tensor_types, ops

slices0 = [slice(0, 63), slice(63, 90), slice(90, 128)]
slices1 = [slice(0, 127), slice(127, 256), slice(256, 257), slice(257, 512)]
AIFB = read_slices_from_csv('AIFB.csv')
AM = read_slices_from_csv('AM.csv')
BGS = read_slices_from_csv('BGS.csv')
DBLP = read_slices_from_csv('DBLP.csv')
MUTAG = read_slices_from_csv('MUTAG.csv')
slices_obj = [("AIFB", AIFB), ("AM", AM), ("BGS", BGS), ("DBLP", DBLP), ("MUTAG", MUTAG)]

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("engine", [Engine.TORCH, Engine.TRITON])
@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("slices", [slices0, slices1, AIFB, AM, BGS, DBLP, MUTAG])
@pytest.mark.parametrize("T", [16, 33])
@pytest.mark.parametrize("K", [16, 32, 64, 80])
def test_segment_matmul(K: int, T: int, slices: list, engine: Engine, device: str, phase: str, dtype: str) -> None:
    if engine == Engine.TRITON and device == "cpu":
        pytest.skip("Triton does not support CPU inference")
    if device == "cpu" and dtype == "float16":
        pytest.skip("CPU does not support FP16")
    dtype = getattr(torch, dtype)
    M = sum([s.stop - s.start for s in slices])
    data = torch.randn((M, K), device=device, dtype=dtype)
    types = torch.zeros((M,), device=device, dtype=torch.int)
    rand_types = torch.randperm(T, device=device, dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = rand_types[i]
    tensor_slice = compact_tensor_types(data, types, device=device)
    other = torch.randn((T, K, K), device=device, dtype=dtype)
    if phase == "forward":
        output = ops.fasten_segment_matmul(tensor_slice.data, other, tensor_slice, engine)
        output_ref = torch.zeros((M, K), device=device, dtype=dtype)
        for i in range(len(tensor_slice)):
            s = tensor_slice.get_slice_from_index(i, is_tensor=False)
            t = tensor_slice.get_type_from_index(i, is_tensor=False)
            output_ref[s] = torch.matmul(tensor_slice.data[s], other[t])
        torch.testing.assert_close(output, output_ref, atol=1e-1, rtol=1e-2)
    elif phase == "backward":
        tensor_slice.data.requires_grad = True
        other.requires_grad = True
        output = ops.fasten_segment_matmul(tensor_slice.data, other, tensor_slice, engine)
        output_grad = torch.randn_like(output)
        output.backward(output_grad)
        sorted_data_grad_ref = torch.zeros_like(data, dtype=dtype)
        other_grad_ref = torch.zeros_like(other, dtype=dtype)
        for i in range(len(tensor_slice)):
            s = tensor_slice.get_slice_from_index(i, is_tensor=False)
            t = tensor_slice.get_type_from_index(i, is_tensor=False)
            sorted_data_grad_ref[s] = torch.matmul(output_grad[s], other[t].t())
            other_grad_ref[t] = torch.matmul(tensor_slice.data[s].t(), output_grad[s])
        torch.testing.assert_close(tensor_slice.data.grad, sorted_data_grad_ref, atol=1e-1, rtol=1e-2)
        if M // T >= 8192:
            # gradient accumlation starts to be significantly different with large samples
            torch.testing.assert_close(other.grad, other_grad_ref, atol=1.0, rtol=1e-2)
        else:
            torch.testing.assert_close(other.grad, other_grad_ref, atol=1e-1, rtol=1e-2)
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def benchmark_results():
    results = []
    yield results

    with open("benchmark_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)
    
    header = results[0].keys()
    with open("benchmark_results.csv", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


@pytest.mark.parametrize("phase", ["forward", "backward", "full"])
@pytest.mark.parametrize("dtype", ["float32"])  # pyg_lib doesn't support float16
@pytest.mark.parametrize("slices_name, slices", slices_obj)
@pytest.mark.parametrize("K", [16, 32, 64])
def test_perf(phase: str, dtype: str, slices_name: str, slices: list, K: int, benchmark_results: Callable[[], None]) -> None:
    T = len(slices)
    M = sum([s.stop - s.start for s in slices])
    dtype = getattr(torch, dtype)
    data = torch.randn((M, K), device="cuda", dtype=dtype)
    types = torch.zeros((M,), device="cuda", dtype=torch.int)
    rand_types = torch.randperm(T, device="cuda", dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = rand_types[i]
    tensor_slice = compact_tensor_types(data, types, device="cuda")
    other = torch.randn((T, K, K), device="cuda", dtype=dtype)
    # ptr should be on CPU
    ptr = torch.tensor([s.start for s in slices] + [slices[-1].stop])

    if phase == "backward" or phase == "full":
        data.requires_grad = True
        other.requires_grad = True

    # warmup and get output
    output_fasten = ops.fasten_segment_matmul(data, other, tensor_slice, Engine.AUTO)
    output_pyg = pyg_lib.ops.segment_matmul(data, ptr, other)
    grad_fasten = torch.empty_like(output_fasten)
    grad_pyg = torch.empty_like(output_pyg)

    def fasten_fn():
        if phase == "full":
            output = ops.fasten_segment_matmul(data, other, tensor_slice, Engine.AUTO)
            output.backward(grad_fasten)
        elif phase == "forward":
            ops.fasten_segment_matmul(data, other, tensor_slice)
        else:  # phase == "backward"
            output_fasten.backward(grad_fasten, retain_graph=True)

    def pyg_fn():
        if phase == "full":
            output = pyg_lib.ops.segment_matmul(data, ptr, other)
            output.backward(grad_pyg)
        elif phase == "forward":
            pyg_lib.ops.segment_matmul(data, ptr, other)
        else:  # phase == "backward"
            output_pyg.backward(grad_pyg, retain_graph=True)

    fasten_ms = triton.testing.do_bench(fasten_fn)
    pyg_ms = triton.testing.do_bench(pyg_fn)
    print(f"{phase}: fasten: {fasten_ms} ms vs pyg: {pyg_ms} ms")

    benchmark_results.append({
        "phase": phase,
        "dataset": slices_name,
        "K": K,
        "fasten_ms": fasten_ms,
        "pyg_ms": pyg_ms
    })


def test_cache():
    M = 128
    K = 16
    T = 16
    data = torch.randn((M, K), device='cuda', dtype=torch.float32)
    types = torch.zeros((M,), device='cuda', dtype=torch.int)
    slices = [slice(0, 63), slice(63, 90), slice(90, 128)]
    for s in slices:
        if s.stop > s.start:
            types[s] = torch.randint(0, T, (s.stop - s.start,), device='cuda', dtype=torch.int)
    tensor_slice = compact_tensor_types(data, types, device='cuda')
    other = torch.randn((T, K, K), device='cuda', dtype=torch.float32)
    ops.fasten_segment_matmul(tensor_slice.data, other, tensor_slice, Engine.TRITON)
    assert len(tensor_slice._cache) == 1
    assert len(tensor_slice._cache['segment_matmul_forward']) == 1