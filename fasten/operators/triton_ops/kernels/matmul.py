import triton
import triton.language as tl


@triton.jit
def _reg_matmul(
    pid_n, type_id,
    start_off,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr
):
    offs_m = start_off + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = other + type_id * stride_other_b + \
        (offs_k[:, None] * stride_other_k + rn[None, :] * stride_other_n)

    acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
    a = tl.zeros((TILE_M, TILE_K), dtype=input.dtype.element_ty)
    b = tl.zeros((TILE_K, TILE_N), dtype=other.dtype.element_ty)

    for i in range(0, BLOCK_SIZE):
        if i == 0:
            a = tl.load(input_ptrs)
            b = tl.load(other_ptrs)
        else:
            a = tl.load(input_ptrs)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += TILE_M * stride_input_m

    mask_n = offs_n[None, :] < N
    acc = acc.to(output.dtype.element_ty)
    c_ptrs = output + stride_output_m * \
        offs_m[:, None] + stride_output_n * offs_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_n)


@triton.jit
def _matmul(
    pid_n, type_id,
    start_off, end_off,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    MASK_M: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_K: tl.constexpr
):
    offs_m = start_off + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = other + type_id * stride_other_b + \
        (offs_k[:, None] * stride_other_k + rn[None, :] * stride_other_n)

    acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
    mask_m = offs_m[:, None] < end_off if MASK_M else True

    k_iter = K // TILE_K if EVEN_K else tl.cdiv(K, TILE_K)
    for k in range(0, k_iter):
        if EVEN_K:
            if MASK_M:
                a = tl.load(input_ptrs, mask=mask_m, other=0.0)
                b = tl.load(other_ptrs)
            else:
                a = tl.load(input_ptrs)
                b = tl.load(other_ptrs)
        else:
            if MASK_M:
                a = tl.load(input_ptrs, mask=mask_m & (offs_k[None, :] + k * TILE_K < K), other=0.0)
                b = tl.load(other_ptrs, mask=(offs_k[:, None] + k * TILE_K < K), other=0.0)
            else:
                a = tl.load(input_ptrs, mask=(offs_k[None, :] + k * TILE_K < K), other=0.0)
                b = tl.load(other_ptrs, mask=(offs_k[:, None] + k * TILE_K < K), other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += TILE_K * stride_input_k
        other_ptrs += TILE_K * stride_other_k

    mask_n = offs_n[None, :] < N
    acc = acc.to(output.dtype.element_ty)
    c_ptrs = output + stride_output_m * \
        offs_m[:, None] + stride_output_n * offs_n[None, :]
    if MASK_M:
        tl.store(c_ptrs, acc, mask=mask_m & mask_n)
    else:
        tl.store(c_ptrs, acc, mask_n)
