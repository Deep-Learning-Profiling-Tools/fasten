from typing import Optional

import torch
import triton
import triton.language as tl

from ...utils import torch_dtype_to_triton_dtype
from .kernels.matmul import (_dynamic_k_matmul, _fast_matmul_inline,
                             _fast_matmul_noinline, _matmul, _reg_matmul)


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
        _matmul(
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
        _matmul(
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
        _matmul(
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
        _matmul(
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
                _dispatch(
                    pid_n,
                    start_off, end_off,
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
    if EQUAL_K and TILE_K <= 32:
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
            cur_start_off = start_off + i * TILE_M
            cur_end_off = cur_start_off + TILE_M
            if EVEN_K and EVEN_N:
                if BLOCK_SIZE == 1 or (TILE_K * TILE_M + TILE_K * TILE_N) <= (64 * 64 * 2):
                    _fast_matmul_inline(
                        cur_start_off, pid_n * TILE_N,
                        input, other + type_id * stride_other_b, output,
                        stride_input_m, stride_input_k,
                        stride_other_k, stride_other_n,
                        stride_output_m, stride_output_n,
                        out_dtype=out_dtype,
                        K_ITER=K // TILE_K,
                        TILE_M=TILE_M,
                        TILE_N=TILE_N,
                        TILE_K=TILE_K
                    )
                else:
                    _fast_matmul_noinline(
                        cur_start_off, pid_n * TILE_N,
                        input, other + type_id * stride_other_b, output,
                        stride_input_m, stride_input_k,
                        stride_other_k, stride_other_n,
                        stride_output_m, stride_output_n,
                        out_dtype=out_dtype,
                        K_ITER=K // TILE_K,
                        TILE_M=TILE_M,
                        TILE_N=TILE_N,
                        TILE_K=TILE_K
                    )
            else:
                _matmul(
                    pid_n,
                    cur_start_off, cur_end_off,
                    input, other + type_id * stride_other_b, output,
                    K, N,
                    stride_input_m, stride_input_k,
                    stride_other_k, stride_other_n,
                    stride_output_m, stride_output_n,
                    out_dtype=out_dtype,
                    MASK_M=False,
                    EVEN_K=EVEN_K,
                    EVEN_N=EVEN_N,
                    TILE_M=TILE_M,
                    TILE_N=TILE_N,
                    TILE_K=TILE_K
                )


@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 128, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 128, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 128, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 128, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['N', 'K'],  # Tune for each N and K, high latency
    # TODO: Employ another performance model
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
    out_dtype: tl.constexpr,
    NUM_TILES: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # A key to determine whether to autotune during training
    BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EQUAL_K: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
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


@triton.jit
def _split_noncontiguous_block(
    pid_k, pid_n,
    input, input_tiles, grad_output, grad_other,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    K, N, next_id, next_next_id,
    out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    for _ in range(0, BLOCK_SIZE):
        if next_id < NUM_BLOCKS and next_id != -1:
            # TODO: large tensors
            # Use int32 to reduce register usage
            start_off = tl.load(input_tiles + 5 * next_id + 2)
            end_off = tl.load(input_tiles + 5 * next_id + 3)
            length = end_off - start_off

            if length > 0:
                type_id = tl.load(input_tiles + 5 * next_id + 1)

                cur_input = input + start_off * stride_input_m
                cur_grad_output = grad_output + start_off * stride_grad_output_m
                cur_grad_other = grad_other + type_id * stride_grad_other_b
                _dynamic_k_matmul(
                    pid_k, pid_n,
                    cur_input, cur_grad_output, cur_grad_other,
                    stride_input_m, stride_input_k,
                    stride_grad_output_m, stride_grad_output_n,
                    stride_grad_other_k, stride_grad_other_n,
                    K, N, length,
                    out_dtype=out_dtype,
                    TILE_K=TILE_K,
                    TILE_N=TILE_N,
                    TILE_M=TILE_M,
                    EVEN_K=EVEN_K,
                    EVEN_N=EVEN_N,
                    EVEN_M=False,
                )
            next_id = next_next_id
            next_next_id += 1


@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 32, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'TILE_SIZE_N': 64, 'TILE_SIZE_K': 32}, num_warps=4, num_stages=4),
    ],
    reset_to_zero=['grad_other'],
    key=['N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['TILE_SIZE_K'] == 0,
    'EVEN_N': lambda args: args['N'] % args['TILE_SIZE_N'] == 0
})
@triton.jit
def split_matmul_kernel(
    input, input_tiles, grad_output, grad_other,
    K, N,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    out_dtype: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr
):
    # TODO(Keren): a different block grouping scheme
    pid = tl.program_id(axis=0)
    grid_k = tl.cdiv(K, TILE_SIZE_K)
    grid_n = tl.cdiv(N, TILE_SIZE_N)
    next_id = pid // (grid_k * grid_n)
    tile_id = pid % (grid_k * grid_n)
    pid_k = tile_id // grid_n
    pid_n = tile_id % grid_n
    next_next_id = tl.load(input_tiles + 5 * next_id + 4)

    # contiguous block
    if next_next_id == 0:
        start_off = tl.load(input_tiles + 5 * next_id + 2)
        type_id = tl.load(input_tiles + 5 * next_id + 1)
        grad_other = grad_other + type_id * stride_grad_other_b
        _dynamic_k_matmul(
            pid_k, pid_n,
            input + start_off * stride_input_m,
            grad_output + start_off * stride_grad_output_m, grad_other,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_k, stride_grad_other_n,
            K, N, TILE_SIZE_M * BLOCK_SIZE,
            out_dtype=out_dtype,
            TILE_K=TILE_SIZE_K,
            TILE_N=TILE_SIZE_N,
            TILE_M=TILE_SIZE_M,
            EVEN_K=EVEN_K,
            EVEN_N=EVEN_N,
            EVEN_M=True,
        )
    else:
        _split_noncontiguous_block(
            pid_k, pid_n,
            input, input_tiles, grad_output, grad_other,
            stride_input_m, stride_input_k,
            stride_grad_output_m, stride_grad_output_n,
            stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
            K, N, next_id, next_next_id,
            out_dtype=out_dtype,
            BLOCK_SIZE=BLOCK_SIZE,
            NUM_BLOCKS=NUM_BLOCKS,
            TILE_K=TILE_SIZE_K,
            TILE_N=TILE_SIZE_N,
            TILE_M=TILE_SIZE_M,
            EVEN_K=EVEN_K,
            EVEN_N=False,
        )


def segment_matmul_forward(input: torch.Tensor, other: torch.Tensor,
                           input_tiles: torch.Tensor, input_slices: torch.Tensor,
                           output: torch.Tensor = None,
                           num_blocks: Optional[int] = None, block_size: int = 1, contiguous_ratio: float = 1.0, tile_size: int = 64, out_dtype: torch.dtype = None):
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
                                  num_blocks: Optional[int] = None, block_size: int = 1, contiguous_ratio: float = 1.0, tile_size: int = 64):
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
                                  num_blocks: Optional[int] = None, block_size: int = 1, contiguous_ratio: float = 1.0, tile_size: int = 64):
    assert input.size(1) == other.size(1)
    assert input_tiles.device == input_slices.device == input.device == other.device
    assert input.dim() == 2
    assert other.dim() == 3
    K: int = input.size(1)
    N: int = other.size(2)
    num_tiles = input_tiles.size(0)
    num_blocks = num_blocks or num_tiles
    grad_output = grad_output.contiguous()

    #  [M, K]^T x [M, N]-> [K, N]
    if grad_other is None:
        # grad_other might be sparse
        grad_other = torch.zeros_like(other)

    def grid(meta):
        return ((num_blocks * triton.cdiv(K, meta['TILE_SIZE_K']) * triton.cdiv(N, meta['TILE_SIZE_N'])), )
    out_dtype = torch_dtype_to_triton_dtype(grad_output.dtype, grad=True)
    split_matmul_kernel[grid](
        input, input_tiles, grad_output, grad_other,
        K, N,
        input.stride(0), input.stride(1),
        grad_output.stride(0), grad_output.stride(1),
        grad_other.stride(0), grad_other.stride(1), grad_other.stride(2),
        out_dtype=out_dtype,
        NUM_BLOCKS=num_blocks,
        TILE_SIZE_M=tile_size,
        BLOCK_SIZE=block_size,
    )
    return grad_other
