from typing import Optional

import torch
import triton
import triton.language as tl

from ...utils import torch_dtype_to_triton_dtype


@triton.jit
def _dynamic_tiling(
    pid_n, type_id,
    start_off, end_off,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr
):
    offs_m = start_off + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = other + type_id * stride_other_b + \
        (offs_k[:, None] * stride_other_k + rn[None, :] * stride_other_n)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=out_dtype)
    mask_m = offs_m[:, None] < end_off

    k_iter = K // BLOCK_K if EVEN_K else tl.cdiv(K, BLOCK_K)
    for k in range(0, k_iter):
        if EVEN_K:
            a = tl.load(input_ptrs, mask=mask_m, other=0.0)
            b = tl.load(other_ptrs)
        else:
            a = tl.load(input_ptrs, mask=mask_m & (offs_k[None, :] + k * BLOCK_K < K), other=0.0)
            b = tl.load(other_ptrs, mask=(offs_k[:, None] + k * BLOCK_K < K), other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += BLOCK_K * stride_input_k
        other_ptrs += BLOCK_K * stride_other_k

    mask_n = offs_n[None, :] < N
    acc = acc.to(output.dtype.element_ty)
    c_ptrs = output + stride_output_m * \
        offs_m[:, None] + stride_output_n * offs_n[None, :]
    c_mask = mask_m & mask_n
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_K'] == 0
})
@triton.jit
def segment_matmul_kernel(
    input, input_tiles, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    other_transposed: tl.constexpr,
    out_dtype: tl.constexpr,
    DYNAMIC_TILING: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,  # it is not used but we need it as a key to differentiate between default and balanced tiling
    EVEN_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    BLOCK_N: tl.constexpr = BLOCK_SIZE_K if other_transposed else BLOCK_SIZE_N
    BLOCK_K: tl.constexpr = BLOCK_SIZE_N if other_transposed else BLOCK_SIZE_K

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    next_id = pid_m

    for j in range(0, 8):
        # TODO: large tensors
        # Use int32 to reduce register usage
        if next_id != -1:
            start_off = tl.load(input_tiles + 5 * next_id + 2).to(tl.int32)
            end_off = tl.load(input_tiles + 5 * next_id + 3).to(tl.int32)
            BLOCK_M_16: tl.constexpr = 16
            BLOCK_M_32: tl.constexpr = 32
            BLOCK_M_64: tl.constexpr = 64

            length = end_off - start_off
            if length > 0:
                type_id = tl.load(input_tiles + 5 * next_id + 1).to(tl.int32)
                for i in range(0, tl.cdiv(length, BLOCK_SIZE_M)):
                    cur_start_off = start_off + i * BLOCK_SIZE_M
                    cur_end_off = min(cur_start_off + BLOCK_SIZE_M, end_off)
                    if cur_end_off - cur_start_off <= BLOCK_M_16 and DYNAMIC_TILING:
                        _dynamic_tiling(
                            pid_n, type_id,
                            cur_start_off, cur_end_off,
                            input, other, output,
                            K, N,
                            stride_input_m, stride_input_k,
                            stride_other_b, stride_other_k, stride_other_n,
                            stride_output_m, stride_output_n,
                            out_dtype=out_dtype,
                            EVEN_K=EVEN_K,
                            BLOCK_M=BLOCK_M_16,
                            BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K
                        )
                    elif cur_end_off - cur_start_off <= BLOCK_M_32 and DYNAMIC_TILING:
                        _dynamic_tiling(
                            pid_n, type_id,
                            cur_start_off, cur_end_off,
                            input, other, output,
                            K, N,
                            stride_input_m, stride_input_k,
                            stride_other_b, stride_other_k, stride_other_n,
                            stride_output_m, stride_output_n,
                            out_dtype=out_dtype,
                            EVEN_K=EVEN_K,
                            BLOCK_M=BLOCK_M_32,
                            BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K
                        )
                    elif cur_end_off - cur_start_off <= BLOCK_M_64 and DYNAMIC_TILING:
                        _dynamic_tiling(
                            pid_n, type_id,
                            cur_start_off, cur_end_off,
                            input, other, output,
                            K, N,
                            stride_input_m, stride_input_k,
                            stride_other_b, stride_other_k, stride_other_n,
                            stride_output_m, stride_output_n,
                            out_dtype=out_dtype,
                            EVEN_K=EVEN_K,
                            BLOCK_M=BLOCK_M_64,
                            BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K
                        )
                    else:
                        _dynamic_tiling(
                            pid_n, type_id,
                            cur_start_off, cur_end_off,
                            input, other, output,
                            K, N,
                            stride_input_m, stride_input_k,
                            stride_other_b, stride_other_k, stride_other_n,
                            stride_output_m, stride_output_n,
                            out_dtype=out_dtype,
                            EVEN_K=EVEN_K,
                            BLOCK_M=BLOCK_SIZE_M,
                            BLOCK_N=BLOCK_N,
                            BLOCK_K=BLOCK_K
                        )
            next_id = tl.load(input_tiles + 5 * next_id + 4).to(tl.int32)


# TODO(Keren): split_matmul_kernel
# We should be able to autotune between split matmul and batch matmul
# with an algorithm selector.
# split matmul uses persistent loops
# batch matmul uses parallel loops


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_M': 32}, num_warps=4, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def batch_matmul_kernel(
    input, input_slices, grad_output, grad_other,
    K, N,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    out_dtype: tl.constexpr,
    B: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # TODO(Keren): a different block grouping scheme
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    bid = tl.program_id(axis=2)

    start_off = tl.load(input_slices + 5 * bid + 2)
    end_off = tl.load(input_slices + 5 * bid + 3)
    if end_off <= start_off:
        return

    type_id = tl.load(input_slices + 5 * bid + 1)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)

    # [M, K] -> [K, M]
    input_ptrs = input + ((offs_m[None, :] + start_off) * stride_input_m + offs_k[:, None] * stride_input_k)
    # [M, N]
    grad_output_ptrs = grad_output + ((offs_m[:, None] + start_off) * stride_grad_output_m + offs_n[None, :] * stride_grad_output_n)

    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=out_dtype)
    mask_k = offs_k[:, None] < K
    mask_n = offs_n[None, :] < N
    M = end_off - start_off

    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        a = tl.load(input_ptrs, mask=mask_k & (offs_m[None, :] + m * BLOCK_SIZE_M < M), other=0.0)
        b = tl.load(grad_output_ptrs, mask=mask_n & (offs_m[:, None] + m * BLOCK_SIZE_M < M), other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += BLOCK_SIZE_M * stride_input_m
        grad_output_ptrs += BLOCK_SIZE_M * stride_grad_output_m

    acc = acc.to(grad_other.dtype.element_ty)
    c_ptrs = grad_other + type_id * stride_grad_other_b + \
        stride_grad_other_k * offs_k[:, None] + stride_grad_other_n * offs_n[None, :]
    c_mask = mask_k & mask_n
    tl.store(c_ptrs, acc, mask=c_mask)


def segment_matmul_forward(input: torch.Tensor, other: torch.Tensor,
                           input_tiles: torch.Tensor, input_slices: torch.Tensor,
                           output: torch.Tensor = None,
                           num_blocks: Optional[int] = None, tile_size: int = 64, out_dtype: torch.dtype = None):
    assert input.size(1) == other.size(1)
    assert input_tiles.device == input_slices.device == input.device == other.device
    assert input.dim() == 2
    assert other.dim() == 3
    M: int = input.size(0)
    K: int = input.size(1)
    N: int = other.size(2)
    num_blocks = num_blocks or input_tiles.size(0)
    if output is None:
        output = torch.empty(M, N, dtype=input.dtype, device=input.device)

    def grid(meta):
        return (num_blocks, triton.cdiv(N, meta['BLOCK_SIZE_N']))
    out_dtype = torch_dtype_to_triton_dtype(out_dtype or input.dtype)
    segment_matmul_kernel[grid](
        input, input_tiles, other, output,
        K, N,
        input.stride(0), input.stride(1),
        other.stride(0), other.stride(1), other.stride(2),
        output.stride(0), output.stride(1),
        DYNAMIC_TILING=False,
        NUM_BLOCKS=num_blocks,
        other_transposed=False,
        out_dtype=out_dtype,
        BLOCK_SIZE_M=tile_size,
    )
    return output


def segment_matmul_backward(input: torch.Tensor, grad_output: torch.Tensor, other: torch.Tensor,
                            input_tiles: torch.Tensor, input_slices: torch.Tensor,
                            grad_other: torch.Tensor = None, grad_input: torch.Tensor = None,
                            num_blocks: Optional[int] = None, tile_size: int = 64):
    assert input.size(1) == other.size(1)
    assert input_tiles.device == input_slices.device == input.device == other.device
    assert input.dim() == 2
    assert other.dim() == 3
    K: int = input.size(1)
    N: int = other.size(2)
    B: int = input_slices.size(0)
    num_blocks = num_blocks or input_tiles.size(0)
    grad_output = grad_output.contiguous()

    def dx(grad_input):
        # [M, N] x [K, N]^T -> [M, K]
        if grad_input is None:
            grad_input = torch.empty_like(input)

        def grid(meta):
            return (num_blocks, triton.cdiv(K, meta['BLOCK_SIZE_K']))
        out_dtype = torch_dtype_to_triton_dtype(grad_output.dtype)
        segment_matmul_kernel[grid](
            grad_output, input_tiles, other, grad_input,
            N, K,
            grad_output.stride(0), grad_output.stride(1),
            other.stride(0), other.stride(2), other.stride(1),  # swap K and N
            grad_input.stride(0), grad_input.stride(1),
            DYNAMIC_TILING=False,
            NUM_BLOCKS=num_blocks,
            other_transposed=True,
            out_dtype=out_dtype,
            BLOCK_SIZE_M=tile_size,
        )
        return grad_input

    def dw(grad_other):
        #  [M, K]^T x [M, N]-> [K, N]
        if grad_other is None:
            # grad_other might be sparse
            grad_other = torch.zeros_like(other)

        def grid(meta):
            return (triton.cdiv(K, meta['BLOCK_SIZE_K']), triton.cdiv(N, meta['BLOCK_SIZE_N']), B)
        out_dtype = torch_dtype_to_triton_dtype(grad_output.dtype, grad=True)
        batch_matmul_kernel[grid](
            input, input_slices, grad_output, grad_other,
            K, N,
            input.stride(0), input.stride(1),
            grad_output.stride(0), grad_output.stride(1),
            grad_other.stride(0), grad_other.stride(1), grad_other.stride(2),
            out_dtype=out_dtype,
            B=B
        )
        return grad_other

    return dx(grad_input), dw(grad_other)
