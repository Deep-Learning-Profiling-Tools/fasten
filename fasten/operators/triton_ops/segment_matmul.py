import functools
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import get_dram_gbps, get_tensorcore_tflops
from triton.runtime import driver

from ...utils import GlobalConfig, binning, is_debug, torch_dtype_to_triton_dtype
from .kernels.matmul import _dynamic_matmul, _general_matmul, _reg_matmul


@triton.jit(noinline=True)
def _dispatch(
    pid_n,
    start_off, end_off,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    MASK_M: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    DYNAMIC_TILING: tl.constexpr
):
    TILE_M_16: tl.constexpr = 16
    TILE_M_32: tl.constexpr = 32
    TILE_M_64: tl.constexpr = 64

    if end_off - start_off <= TILE_M_16 and DYNAMIC_TILING:
        _general_matmul(
            pid_n,
            start_off, end_off,
            input, other, output,
            K, N,
            stride_input_m, stride_input_k,
            stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            MASK_M=True,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            TILE_M=TILE_M_16,
            TILE_N=TILE_N,
            TILE_K=TILE_K
        )
    elif end_off - start_off <= TILE_M_32 and DYNAMIC_TILING:
        _general_matmul(
            pid_n,
            start_off, end_off,
            input, other, output,
            K, N,
            stride_input_m, stride_input_k,
            stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            MASK_M=True,
            TILE_M=TILE_M_32,
            TILE_N=TILE_N,
            TILE_K=TILE_K
        )
    elif end_off - start_off <= TILE_M_64 and DYNAMIC_TILING:
        _general_matmul(
            pid_n,
            start_off, end_off,
            input, other, output,
            K, N,
            stride_input_m, stride_input_k,
            stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            MASK_M=True,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            TILE_M=TILE_M_64,
            TILE_N=TILE_N,
            TILE_K=TILE_K
        )
    else:
        _general_matmul(
            pid_n,
            start_off, end_off,
            input, other, output,
            K, N,
            stride_input_m, stride_input_k,
            stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            MASK_M=MASK_M,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K
        )


@triton.jit
def _noncontiguous_block(
    input_tiles,
    next_id, next_next_id, pid_n,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr
):
    for _ in range(0, BLOCK_SIZE):
        if next_id < NUM_TILES and next_id != -1:
            # TODO: large tensors
            # Use int32 to reduce register usage
            start_off = tl.load(input_tiles + 5 * next_id + 2)
            end_off = tl.load(input_tiles + 5 * next_id + 3)
            length = end_off - start_off

            if length > 0:
                type_id = tl.load(input_tiles + 5 * next_id + 1)
                for i in range(0, tl.cdiv(length, TILE_M)):
                    _dispatch(
                        pid_n,
                        start_off + i * TILE_M, end_off,
                        input, other + type_id * stride_other_b, output,
                        K, N,
                        stride_input_m, stride_input_k,
                        stride_other_k, stride_other_n,
                        stride_output_m, stride_output_n,
                        out_dtype=out_dtype,
                        MASK_M=True,
                        EVEN_K=EVEN_K,
                        EVEN_N=EVEN_N,
                        TILE_M=TILE_M,
                        TILE_N=TILE_N,
                        TILE_K=TILE_K,
                        DYNAMIC_TILING=True,
                    )
            next_id = next_next_id
            next_next_id += 1


@triton.jit
def _contiguous_block(
    input_tiles,
    next_id, pid_n,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EQUAL_K: tl.constexpr,
):
    start_off = tl.load(input_tiles + 5 * next_id + 2)
    type_id = tl.load(input_tiles + 5 * next_id + 1)
    if EQUAL_K:
        _reg_matmul(
            pid_n, type_id,
            start_off,
            input, other, output, N,
            stride_input_m, stride_input_k,
            stride_other_b, stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            EVEN_N=EVEN_N,
        )
    else:
        for i in range(0, BLOCK_SIZE):
            _general_matmul(
                pid_n,
                start_off + i * TILE_M,
                start_off + (i + 1) * TILE_M,
                input, other + type_id * stride_other_b, output,
                K, N,
                stride_input_m, stride_input_k,
                stride_other_k, stride_other_n,
                stride_output_m, stride_output_n,
                out_dtype=out_dtype,
                MASK_M=True,
                EVEN_K=EVEN_K,
                EVEN_N=EVEN_N,
                TILE_M=TILE_M,
                TILE_N=TILE_N,
                TILE_K=TILE_K
            )


def _early_config_prune(configs: triton.Config, named_args: dict, is_weight: bool, **kwargs):
    if not GlobalConfig.with_autotune:
        return [configs[0]]
    pruned_configs = []
    element_size = named_args['input'].element_size()
    N = named_args['N']
    K = named_args['K']
    TILE_SIZE_M = kwargs['TILE_SIZE_M']
    BLOCK_SIZE = kwargs['BLOCK_SIZE']
    device = torch.cuda.current_device()
    min_tile_size_n = min([config.kwargs['TILE_SIZE_N'] for config in configs])
    min_tile_size_k = min([config.kwargs['TILE_SIZE_K'] for config in configs])
    max_shared_memory = driver.active.utils.get_device_properties(device)["max_shared_mem"]
    for config in configs:
        kw = config.kwargs
        TILE_SIZE_N = kw['TILE_SIZE_N']
        TILE_SIZE_K = kw['TILE_SIZE_K']
        if is_weight:
            # 1. Prune too large tiles
            if ((TILE_SIZE_K > K and TILE_SIZE_K != min_tile_size_k) or (TILE_SIZE_N > N and TILE_SIZE_N != min_tile_size_n)):
                print(f"Pruned 1: {config}")
                continue
            # 2. Prune configs by shared memory usage
            required_shared_memory = (TILE_SIZE_K + TILE_SIZE_N) * TILE_SIZE_M * config.num_stages * element_size
            if required_shared_memory > max_shared_memory:
                print(f"Pruned 2: {config}")
                continue
            # 3. Prune configs with large tile sizes and small warp sizes (register pressure)
            if TILE_SIZE_K >= 256 and TILE_SIZE_N >= 256 and config.num_warps == 4:
                print(f"Pruned 3: {config}")
                continue
            # 4. Prune M dimension by only eliminating the following three cases:
            # Too many stages
            if TILE_SIZE_M * (config.num_stages - 1) < BLOCK_SIZE * TILE_SIZE_M:
                continue
        else:
            # 1. Prune configs that use more registers and shared memory than necessary
            if TILE_SIZE_N > N and TILE_SIZE_N != min_tile_size_n:
                continue
            # 2. Prune configs by shared memory usage
            required_shared_memory = (TILE_SIZE_M + TILE_SIZE_N) * TILE_SIZE_K * config.num_stages * element_size
            if required_shared_memory > max_shared_memory:
                continue
            # 3. Prune configs with large tile sizes and small warp sizes (register pressure)
            if TILE_SIZE_N >= 256 and TILE_SIZE_K >= 256 and config.num_warps == 4:
                continue
            # 4. Prune K dimension by only eliminating the following three cases:
            # Not register blocking, too many stages, or too few stages
            if TILE_SIZE_K != K and (TILE_SIZE_K * (config.num_stages - 1) > K or TILE_SIZE_K * (config.num_stages + 1) < K):
                continue
            # 5. Large K don't use register blocking
            if TILE_SIZE_K == K and K >= 128:
                continue
        print(f"Kept: {config}")
        pruned_configs.append(config)
    if is_debug():
        print(f"Number of configs pruned from {len(configs)} to {len(pruned_configs)}, is_weight={is_weight}")
    return pruned_configs


def _perf_model(
        input, input_tiles, other, output,
        K, N,
        stride_input_m, stride_input_k,
        stride_other_b, stride_other_k, stride_other_n,
        stride_output_m, stride_output_n,
        stddev_tile_size_m,
        avg_tile_size_m,
        out_dtype: tl.constexpr,
        NUM_TILES: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_SIZE_M: tl.constexpr,
        TILE_SIZE_N: tl.constexpr,
        TILE_SIZE_K: tl.constexpr,
        num_stages, num_warps, **kwargs):
    if not GlobalConfig.with_perf_model:
        return 1.0
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    num_sm_map = {80: 108, 89: 128, 90: 114}
    threads_sm_map = {80: 2048, 89: 1536, 90: 2048}
    cap = capability[0] * 10 + capability[1]
    if cap not in num_sm_map:
        # Unknown architecture
        return 1.0
    # Parallel efficiency
    sms = num_sm_map[cap]
    threads_per_sm = threads_sm_map[cap]
    element_size = input.element_size()
    dtype = input.dtype()
    max_shared_memory = driver.active.utils.get_device_properties(device)["max_shared_mem"]
    required_shared_memory = (TILE_SIZE_M + TILE_SIZE_N) * TILE_SIZE_K * num_stages * element_size
    num_ctas_per_sm = min(max_shared_memory // required_shared_memory, threads_per_sm // (num_warps * 32))
    num_ctas = NUM_BLOCKS * triton.cdiv(N, TILE_SIZE_N)
    ctas_per_wave = num_ctas_per_sm * sms
    parallel_efficiency = num_ctas / (triton.cdiv(num_ctas, ctas_per_wave) * ctas_per_wave)
    # Compute efficiency
    # 1. Compute
    ops = TILE_SIZE_M * TILE_SIZE_N * TILE_SIZE_K * 2
    tensorcore_tflops = get_tensorcore_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = ops / tensorcore_tflops
    # 2. Indexing
    estimated_l2_latency = 200  # TODO: Fix
    estimated_gld_throughput = 500  # TODO: Fix
    indexing_ms = NUM_BLOCKS * (0.1 * estimated_gld_throughput * 1e-3 + 0.9 * estimated_l2_latency * 1e-3)
    # 3. Sync
    estimated_sync_latency = 50  # TODO: Fix
    num_iters = triton.cdiv(K, TILE_SIZE_K)
    sync_ms = num_iters * estimated_sync_latency * 1e-3 * num_warps // 32
    # 4. Store
    store_bytes = TILE_SIZE_M * TILE_SIZE_N * element_size
    estimated_l2_bw = 5 * 1e3
    store_ms = store_bytes / estimated_l2_bw
    # 5. Load
    dram_bw = get_dram_gbps(device)
    load_bytes = (TILE_SIZE_M + TILE_SIZE_N) * TILE_SIZE_K * element_size * BLOCK_SIZE
    load_ms = load_bytes / (0.6 * dram_bw + 0.4 * estimated_l2_bw)
    compute_efficiency = compute_ms / (compute_ms + indexing_ms + sync_ms + store_ms + load_ms)
    # Only prune those with both low parallel and compute efficiency
    return min(1 - parallel_efficiency, 1 - compute_efficiency)


def _generate_configs():
    tile_sizes = [32, 64, 128, 256]  # Possible values for TILE_SIZE_N and TILE_SIZE_K
    num_warps_options = [4, 8]  # Possible values for num_warps
    num_stages_options = [3, 4]  # Possible values for num_stages

    configs = []
    for tile_size_n in tile_sizes:
        for tile_size_k in tile_sizes:
            for num_warps in num_warps_options:
                for num_stages in num_stages_options:
                    config = triton.Config({'TILE_SIZE_N': tile_size_n, 'TILE_SIZE_K': tile_size_k},
                                           num_warps=num_warps, num_stages=num_stages)
                    configs.append(config)

    return configs


@triton.autotune(
    configs=_generate_configs(),
    key=['N', 'K', 'stddev_tile_size_m', 'avg_tile_size_m'],  # Tune for each N and K, high latency
    prune_configs_by={
        'early_config_prune': functools.partial(_early_config_prune, is_weight=False),
        'perf_model': _perf_model,
        'top_k': 100 if GlobalConfig.with_perf_model else 10,
    },
    rep=10,
    use_cuda_graph=True,
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['TILE_SIZE_K'] == 0,
    'EVEN_N': lambda args: args['N'] % args['TILE_SIZE_N'] == 0,
    'EQUAL_K': lambda args: args['K'] == args['TILE_SIZE_K']
})
@triton.jit
def segment_matmul_kernel(
    input, input_tiles, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    stddev_tile_size_m,
    avg_tile_size_m,
    out_dtype: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EQUAL_K: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr
):
    TILE_N: tl.constexpr = TILE_SIZE_N
    TILE_K: tl.constexpr = TILE_SIZE_K
    TILE_M: tl.constexpr = TILE_SIZE_M

    GROUP_M: tl.constexpr = 4

    # Global grouping
    pid = tl.program_id(axis=0)
    grid_n = tl.cdiv(N, TILE_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(NUM_BLOCKS - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    next_id = pid_m
    next_next_id = tl.load(input_tiles + 5 * next_id + 4)
    if next_next_id == 0:
        _contiguous_block(
            input_tiles,
            next_id, pid_n,
            input, other, output,
            K, N,
            stride_input_m, stride_input_k,
            stride_other_b, stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            BLOCK_SIZE=BLOCK_SIZE,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EQUAL_K=EQUAL_K,
        )
    else:
        _noncontiguous_block(
            input_tiles,
            next_id, next_next_id, pid_n,
            input, other, output,
            K, N,
            stride_input_m, stride_input_k,
            stride_other_b, stride_other_k, stride_other_n,
            stride_output_m, stride_output_n,
            out_dtype=out_dtype,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TILES=NUM_TILES,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N)


@triton.jit(noinline=True)
def _split_dispatch(
    pid_k, pid_n, next_id,
    input, grad_output, grad_other, grad_other_tiles,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    K, N, M, length,
    out_dtype: tl.constexpr,
    BLOCK_LENGTH: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    DYNAMIC_TILING: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
):
    TILE_M_16: tl.constexpr = 16
    TILE_M_32: tl.constexpr = 32
    TILE_M_64: tl.constexpr = 64

    if length <= TILE_M_16 and DYNAMIC_TILING:
        _dynamic_matmul(
            pid_k, pid_n, next_id,
            input, grad_output, grad_other, grad_other_tiles,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, M, length,
            out_dtype=out_dtype,
            BLOCK_LENGTH=BLOCK_LENGTH,
            TILE_K=TILE_K,
            TILE_N=TILE_N,
            TILE_M=TILE_M_16,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EVEN_M=EVEN_M,
            DETERMINISTIC=DETERMINISTIC,
        )
    elif length <= TILE_M_32 and DYNAMIC_TILING:
        _dynamic_matmul(
            pid_k, pid_n, next_id,
            input, grad_output, grad_other, grad_other_tiles,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, M, length,
            out_dtype=out_dtype,
            BLOCK_LENGTH=BLOCK_LENGTH,
            TILE_K=TILE_K,
            TILE_N=TILE_N,
            TILE_M=TILE_M_32,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EVEN_M=EVEN_M,
            DETERMINISTIC=DETERMINISTIC,
        )
    elif length <= TILE_M_64 and DYNAMIC_TILING:
        _dynamic_matmul(
            pid_k, pid_n, next_id,
            input, grad_output, grad_other, grad_other_tiles,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, M, length,
            out_dtype=out_dtype,
            BLOCK_LENGTH=BLOCK_LENGTH,
            TILE_K=TILE_K,
            TILE_N=TILE_N,
            TILE_M=TILE_M_64,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EVEN_M=EVEN_M,
            DETERMINISTIC=DETERMINISTIC,
        )
    else:
        _dynamic_matmul(
            pid_k, pid_n, next_id,
            input, grad_output, grad_other, grad_other_tiles,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, M, length,
            out_dtype=out_dtype,
            BLOCK_LENGTH=BLOCK_LENGTH,
            TILE_K=TILE_K,
            TILE_N=TILE_N,
            TILE_M=TILE_M,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EVEN_M=False,
            DETERMINISTIC=DETERMINISTIC,
        )


@triton.jit
def _split_noncontiguous_block(
    pid_k, pid_n,
    input, input_slices, input_tiles,
    grad_output, grad_other, grad_other_tiles,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    K, N, next_id, next_next_id,
    out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
):
    for _ in range(0, BLOCK_SIZE):
        if next_id < NUM_TILES and next_id != -1:
            # TODO: large tensors
            # Use int32 to reduce register usage
            start_off = tl.load(input_tiles + 5 * next_id + 2)
            end_off = tl.load(input_tiles + 5 * next_id + 3)
            length = end_off - start_off

            if length > 0:
                type_id = tl.load(input_tiles + 5 * next_id + 1)
                slice_id = tl.load(input_tiles + 5 * next_id + 0)
                slice_start = tl.load(input_slices + 5 * slice_id + 2)
                slice_end = tl.load(input_slices + 5 * slice_id + 3)
                M = slice_end - slice_start

                _split_dispatch(
                    pid_k, pid_n, next_id,
                    input + start_off * stride_input_m,
                    grad_output + start_off * stride_grad_output_m,
                    grad_other + type_id * stride_grad_other_b,
                    grad_other_tiles,
                    stride_input_m, stride_input_k,
                    stride_grad_output_m, stride_grad_output_n,
                    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
                    K, N, M, length,
                    out_dtype=out_dtype,
                    BLOCK_LENGTH=TILE_M * BLOCK_SIZE,
                    TILE_K=TILE_K,
                    TILE_N=TILE_N,
                    TILE_M=TILE_M,
                    EVEN_K=EVEN_K,
                    EVEN_N=EVEN_N,
                    EVEN_M=False,
                    DYNAMIC_TILING=True,
                    DETERMINISTIC=DETERMINISTIC,
                )
            next_id = next_next_id
            next_next_id += 1


@triton.autotune(
    configs=_generate_configs(),
    reset_to_zero=['grad_other'],
    key=['N', 'K', 'stddev_tile_size_m', 'avg_tile_size_m'],
    prune_configs_by={
        'early_config_prune': functools.partial(_early_config_prune, is_weight=True)
    },
    use_cuda_graph=True,
    rep=20,  # If longer than 20ms, one iteration might be accurate enough
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['TILE_SIZE_K'] == 0,
    'EVEN_N': lambda args: args['N'] % args['TILE_SIZE_N'] == 0
})
@triton.jit
def split_matmul_kernel(
    input, input_slices, input_tiles,
    grad_output, grad_other, grad_other_tiles,
    K, N,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    stddev_tile_size_m,
    avg_tile_size_m,
    out_dtype: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    DETERMINISTIC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_k = tl.cdiv(K, TILE_SIZE_K)
    grid_n = tl.cdiv(N, TILE_SIZE_N)
    next_id = pid // (grid_k * grid_n)
    next_next_id = tl.load(input_tiles + 5 * next_id + 4)
    tile_id = pid % (grid_k * grid_n)
    pid_k = tile_id // grid_n
    pid_n = tile_id % grid_n

    # contiguous block
    if next_next_id == 0:
        slice_id = tl.load(input_tiles + 5 * next_id + 0)
        type_id = tl.load(input_tiles + 5 * next_id + 1)
        start_off = tl.load(input_tiles + 5 * next_id + 2)
        slice_start = tl.load(input_slices + 5 * slice_id + 2)
        slice_end = tl.load(input_slices + 5 * slice_id + 3)
        M = slice_end - slice_start
        _dynamic_matmul(
            pid_k, pid_n, next_id,
            input + start_off * stride_input_m,
            grad_output + start_off * stride_grad_output_m,
            grad_other + type_id * stride_grad_other_b,
            grad_other_tiles,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, M, TILE_SIZE_M * BLOCK_SIZE,
            out_dtype=out_dtype,
            BLOCK_LENGTH=TILE_SIZE_M * BLOCK_SIZE,
            TILE_K=TILE_SIZE_K,
            TILE_N=TILE_SIZE_N,
            TILE_M=TILE_SIZE_M,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EVEN_M=True,
            DETERMINISTIC=DETERMINISTIC,
        )
    else:
        _split_noncontiguous_block(
            pid_k, pid_n,
            input, input_slices, input_tiles,
            grad_output, grad_other, grad_other_tiles,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, next_id, next_next_id,
            out_dtype=out_dtype,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_TILES=NUM_TILES,
            TILE_K=TILE_SIZE_K,
            TILE_N=TILE_SIZE_N,
            TILE_M=TILE_SIZE_M,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            DETERMINISTIC=DETERMINISTIC,
        )


def _generate_reduce_configs():
    tile_sizes = [32, 64]  # Possible values for TILE_SIZE_N and TILE_SIZE_K
    num_warps_options = [4, 8]  # Possible values for num_warps
    num_stages_options = [1]  # Possible values for num_stages

    configs = []
    for tile_size_n in tile_sizes:
        for tile_size_k in tile_sizes:
            for num_warps in num_warps_options:
                for num_stages in num_stages_options:
                    config = triton.Config({'TILE_SIZE_N': tile_size_n, 'TILE_SIZE_K': tile_size_k},
                                           num_warps=num_warps, num_stages=num_stages)
                    configs.append(config)

    return configs


@triton.autotune(
    configs=_generate_reduce_configs(),
    key=['N', 'K'],
    rep=1,
    use_cuda_graph=True
)
@triton.jit
def split_reduce_kernel(
    slice_to_tiles, grad_other_tiles, grad_other,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    K, N,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_k = tl.cdiv(K, TILE_SIZE_K)
    grid_n = tl.cdiv(N, TILE_SIZE_N)
    slice_id = pid // (grid_k * grid_n)
    pid_k = (pid % (grid_k * grid_n)) // grid_n
    pid_n = (pid % (grid_k * grid_n)) % grid_n
    type_id = tl.load(slice_to_tiles + slice_id * 3 + 0)
    start_tile_id = tl.load(slice_to_tiles + slice_id * 3 + 1)
    end_tile_id = tl.load(slice_to_tiles + slice_id * 3 + 2)
    if start_tile_id == end_tile_id or end_tile_id - start_tile_id == 1:
        return
    acc = tl.zeros((TILE_SIZE_K, TILE_SIZE_N), dtype=grad_other.dtype.element_ty)
    k_offs = pid_k * TILE_SIZE_K + tl.arange(0, TILE_SIZE_K)[:, None]
    n_offs = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)[None, :]
    grad_other_tiles_ptrs = grad_other_tiles + k_offs * stride_grad_other_k + n_offs * stride_grad_other_n
    mask = (k_offs < K) & (n_offs < N)
    for i in range(start_tile_id, end_tile_id):
        acc += tl.load(grad_other_tiles_ptrs + stride_grad_other_b * i, mask=mask)
    tl.store(grad_other + type_id * stride_grad_other_b + k_offs * stride_grad_other_k + n_offs * stride_grad_other_n, acc, mask=mask)


def segment_matmul_forward(input: torch.Tensor, other: torch.Tensor,
                           input_tiles: torch.Tensor, input_slices: torch.Tensor,
                           output: torch.Tensor = None,
                           num_blocks: Optional[int] = None, block_size: int = 1,
                           tile_size: int = 64, out_dtype: Optional[torch.dtype] = None,
                           avg_tile_size: Optional[float] = None, stddev_tile_size: Optional[float] = None, **kwargs):
    assert input.size(1) == other.size(1)
    assert input_tiles.device == input_slices.device == input.device == other.device
    assert input.dim() == 2
    assert other.dim() == 3
    M: int = input.size(0)
    K: int = input.size(1)
    N: int = other.size(2)
    num_tiles = input_tiles.size(0)
    num_blocks = num_blocks or num_tiles
    if output is None:
        output = torch.empty(M, N, dtype=input.dtype, device=input.device)

    def grid(meta):
        return (num_blocks * triton.cdiv(N, meta['TILE_SIZE_N']),)

    out_dtype = torch_dtype_to_triton_dtype(out_dtype or input.dtype)
    segment_matmul_kernel[grid](
        input, input_tiles, other, output,
        K, N,
        input.stride(0), input.stride(1),
        other.stride(0), other.stride(1), other.stride(2),
        output.stride(0), output.stride(1),
        binning(stddev_tile_size, 32),
        binning(avg_tile_size, 16),
        NUM_TILES=num_tiles,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
        out_dtype=out_dtype,
        TILE_SIZE_M=tile_size,
    )
    return output


def segment_matmul_backward_input(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor,
                                  input_tiles: torch.Tensor, input_slices: torch.Tensor,
                                  grad_input: torch.Tensor = None,
                                  num_blocks: Optional[int] = None, block_size: int = 1, tile_size: int = 64,
                                  avg_tile_size: Optional[float] = None, stddev_tile_size: Optional[float] = None, **kwargs):
    assert input_tiles.device == input_slices.device == other.device
    assert other.dim() == 3
    K: int = other.size(1)
    N: int = other.size(2)
    num_tiles = input_tiles.size(0)
    num_blocks = num_blocks or num_tiles
    grad_output = grad_output.contiguous()

    # [M, N] x [K, N]^T -> [M, K]
    if grad_input is None:
        grad_input = torch.empty_like(input)

    def grid(meta):
        return (num_blocks * triton.cdiv(K, meta['TILE_SIZE_N']),)
    out_dtype = torch_dtype_to_triton_dtype(grad_output.dtype)
    segment_matmul_kernel[grid](
        grad_output, input_tiles, other, grad_input,
        N, K,
        grad_output.stride(0), grad_output.stride(1),
        other.stride(0), other.stride(2), other.stride(1),  # swap K and N
        grad_input.stride(0), grad_input.stride(1),
        binning(stddev_tile_size, 32),
        binning(avg_tile_size, 16),
        NUM_TILES=num_tiles,
        NUM_BLOCKS=num_blocks,
        BLOCK_SIZE=block_size,
        out_dtype=out_dtype,
        TILE_SIZE_M=tile_size,
    )
    return grad_input


def segment_matmul_backward_other(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor,
                                  input_tiles: torch.Tensor, input_slices: torch.Tensor,
                                  grad_other: torch.Tensor = None,
                                  num_blocks: Optional[int] = None, block_size: int = 1, tile_size: int = 64,
                                  deterministic: bool = False, slice_tile_mapping: torch.Tensor = None,
                                  avg_tile_size: Optional[float] = None, stddev_tile_size: Optional[float] = None, **kwargs):
    assert input.size(1) == other.size(1)
    assert input_tiles.device == input_slices.device == input.device == other.device
    assert input.dim() == 2
    assert other.dim() == 3
    K: int = input.size(1)
    N: int = other.size(2)
    num_tiles = input_tiles.size(0)
    num_slices = input_slices.size(0)
    num_blocks = num_blocks or num_tiles
    grad_output = grad_output.contiguous()

    #  [M, K]^T x [M, N]-> [K, N]
    if grad_other is None:
        # grad_other might be sparse
        grad_other = torch.zeros_like(other)

    grad_other_tiles = None
    if deterministic:
        grad_other_tiles = torch.zeros((num_tiles, K, N), device=grad_other.device, dtype=grad_other.dtype)

    def grid(meta):
        return ((num_blocks * triton.cdiv(K, meta['TILE_SIZE_K']) * triton.cdiv(N, meta['TILE_SIZE_N'])), )
    out_dtype = torch_dtype_to_triton_dtype(grad_output.dtype, grad=True)
    split_matmul_kernel[grid](
        input, input_slices, input_tiles,
        grad_output, grad_other, grad_other_tiles,
        K, N,
        input.stride(0), input.stride(1),
        grad_output.stride(0), grad_output.stride(1),
        grad_other.stride(0), grad_other.stride(1), grad_other.stride(2),
        binning(stddev_tile_size, 32),
        binning(avg_tile_size, 16),
        out_dtype=out_dtype,
        NUM_BLOCKS=num_blocks,
        NUM_TILES=num_tiles,
        TILE_SIZE_M=tile_size,
        BLOCK_SIZE=block_size,
        DETERMINISTIC=deterministic,
    )
    if deterministic:
        def grid(meta):
            return (num_slices * triton.cdiv(K, meta['TILE_SIZE_K']) * triton.cdiv(N, meta['TILE_SIZE_N']), )
        split_reduce_kernel[grid](
            slice_tile_mapping, grad_other_tiles, grad_other,
            grad_other.stride(0), grad_other.stride(1), grad_other.stride(2),
            K, N
        )
    return grad_other
