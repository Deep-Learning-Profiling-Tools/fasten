import csv
import json
from typing import Callable

import pyg_lib
import pytest
import torch
import triton
from utils import read_slices_from_csv

from fasten import Engine, compact_tensor_types, ops
from fasten.stats import get_matmul_flops

slices0 = [slice(0, 63), slice(63, 90), slice(90, 128)]
slices1 = [slice(0, 127), slice(127, 256), slice(256, 257), slice(257, 512)]
AIFB = read_slices_from_csv('AIFB.csv')
AM = read_slices_from_csv('AM.csv')
BGS = read_slices_from_csv('BGS.csv')
MUTAG = read_slices_from_csv('MUTAG.csv')
slices_obj = [("AIFB", AIFB), ("AM", AM), ("BGS", BGS), ("MUTAG", MUTAG)]

# non-cudagraph tests are not stable on GPU, but pyg_lib only supports the cudagraph mode
use_cudagraph = False


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("engine", [Engine.TORCH, Engine.TRITON])
@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("slices", [slices0, slices1, AIFB, AM, BGS, MUTAG])
@pytest.mark.parametrize("K", [16, 32, 64, 80])
def test_segment_matmul(K: int, slices: list, engine: Engine, device: str, phase: str, dtype: str) -> None:
    if engine == Engine.TRITON and device == "cpu":
        pytest.skip("Triton does not support CPU inference")
    if device == "cpu" and dtype == "float16":
        pytest.skip("CPU does not support FP16")
    T = len(slices)
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
        if M // T >= 2048:
            # gradient accumlation starts to be significantly different with large samples
            torch.testing.assert_close(other.grad, other_grad_ref, atol=1.0, rtol=1e-2)
        else:
            torch.testing.assert_close(other.grad, other_grad_ref, atol=1e-1, rtol=1e-2)
    if device == "cuda":
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def benchmark_results(format: str = "csv"):
    results = []
    yield results

    if format == "json":
        with open("benchmark_results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
    elif format == "csv":
        header = results[0].keys()
        with open("benchmark_results.csv", "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=header)
            writer.writeheader()
            for row in results:
                writer.writerow(row)


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("engine", ["fasten", "pyg", "torch"])
@pytest.mark.parametrize("slices_name, slices", slices_obj)
@pytest.mark.parametrize("K", [32, 64, 128])
def test_perf(phase: str, dtype: str, engine: str, slices_name: str, slices: list, K: int, benchmark_results: Callable[[], None]) -> None:
    if engine == "pyg" and dtype == "float16":
        pytest.skip("pyg_lib does not support float16")
    torch.backends.cuda.matmul.allow_tf32 = True
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

    if phase == "backward":
        data.requires_grad = True
        other.requires_grad = True

    # warmup and get output
    if engine == "fasten":
        output = ops.fasten_segment_matmul(data, other, tensor_slice, Engine.AUTO)
    elif engine == "pyg":
        output = pyg_lib.ops.segment_matmul(data, ptr, other)
    elif engine == "torch":
        output = ops.fasten_segment_matmul(data, other, tensor_slice, Engine.TORCH)

    if phase == "backward":
        grad = torch.randn_like(output)
        if engine == "pyg":
            grouped_data = []
            grouped_grad = []
            for s in slices:
                if s.stop > s.start:
                    grouped_data.append(data[s.start:s.stop, :].t())
                    grouped_grad.append(grad[s.start:s.stop, :])

    def fasten_fn():
        if phase == "forward":
            ops.fasten_segment_matmul(data, other, tensor_slice, Engine.AUTO if engine == "fasten" else Engine.TORCH)
        else:  # phase == "backward"
            output.backward(grad, retain_graph=True)

    def pyg_fn():
        if phase == "forward":
            pyg_lib.ops.segment_matmul(data, ptr, other)
        else:  # phase == "backward"
            # dx
            # [M, N] * [K, N]^T = [M, K]^T
            pyg_lib.ops.segment_matmul(grad, ptr, other.transpose(1, 2))
            # dw
            # [M, K]^T * [M, N] = [K, N]
            pyg_lib.ops.grouped_matmul(grouped_data, grouped_grad)

    fn = pyg_fn if engine == "pyg" else fasten_fn
    ms = triton.testing.do_bench(fn, grad_to_none=[data, other])
    tflops = get_matmul_flops(tensor_slice, other) / 1e12
    print(f"{phase} ms {ms} tflops {tflops}")

    benchmark_results.append({
        "engine": engine,
        "phase": phase,
        "dataset": slices_name,
        "K": K,
        "ms": ms,
    })


@pytest.mark.parametrize("phase", ["forward", "backward"])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("engine", ["fasten", "pyg"])
@pytest.mark.parametrize("K", [32, 128])
@pytest.mark.parametrize("T", [i for i in range(100, 2000, 200)])
@pytest.mark.parametrize("M", [1000000])
def test_perf_random(phase: str, dtype: str, engine: str, K: int, T: int, M: int, benchmark_results: Callable[[], None]):
    if engine == "pyg" and dtype == "float16":
        pytest.skip("pyg_lib does not support float16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.random.manual_seed(147)
    dtype = getattr(torch, dtype)
    data = torch.randn((M, K), device="cuda", dtype=dtype)
    types = torch.zeros((M,), device="cuda", dtype=torch.int)
    types = torch.randint(0, T, (M,), device="cuda", dtype=torch.int)
    tensor_slice = compact_tensor_types(data, types, device="cuda")
    other = torch.randn((T, K, K), device="cuda", dtype=dtype)
    # ptr should be on CPU
    ptr = []
    for i in range(len(tensor_slice)):
        ptr.append(tensor_slice.get_slice_from_index(i, is_tensor=False).start)
    ptr.append(tensor_slice.get_slice_from_index(len(tensor_slice) - 1, is_tensor=False).stop)
    ptr = torch.tensor(ptr)

    if phase == "backward":
        data.requires_grad = True
        other.requires_grad = True

    # warmup and get output
    if engine == "fasten":
        output = ops.fasten_segment_matmul(data, other, tensor_slice, Engine.AUTO)
    elif engine == "pyg":
        output = pyg_lib.ops.segment_matmul(data, ptr, other)
    elif engine == "torch":
        output = ops.fasten_segment_matmul(data, other, tensor_slice, Engine.TORCH)

    if phase == "backward":
        grad = torch.randn_like(output)
        if engine == "pyg":
            grouped_data = []
            grouped_grad = []
            for i in range(len(tensor_slice)):
                s = tensor_slice.get_slice_from_index(i, is_tensor=False)
                if s.stop > s.start:
                    grouped_data.append(data[s.start:s.stop, :].t())
                    grouped_grad.append(grad[s.start:s.stop, :])

    def fasten_fn():
        if phase == "forward":
            ops.fasten_segment_matmul(data, other, tensor_slice, Engine.AUTO if engine == "fasten" else Engine.TORCH)
        else:  # phase == "backward"
            output.backward(grad, retain_graph=True)

    def pyg_fn():
        if phase == "forward":
            pyg_lib.ops.segment_matmul(data, ptr, other)
        else:  # phase == "backward"
            # dx
            # [M, N] * [K, N]^T = [M, K]^T
            pyg_lib.ops.segment_matmul(grad, ptr, other.transpose(1, 2))
            # dw
            # [M, K]^T * [M, N] = [K, N]
            pyg_lib.ops.grouped_matmul(grouped_data, grouped_grad)

    fn = pyg_fn if engine == "pyg" else fasten_fn
    ms = triton.testing.do_bench(fn, grad_to_none=[data, other])
    tflops = get_matmul_flops(tensor_slice, other) / 1e12
    print(f"{phase} ms {ms} tflops {tflops}")

    benchmark_results.append({
        "engine": engine,
        "phase": phase,
        "T": T,
        "K": K,
        "ms": ms,
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
