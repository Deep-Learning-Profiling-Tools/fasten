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
    EVEN_N: tl.constexpr,
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

    acc = acc.to(output.dtype.element_ty)
    c_ptrs = output + stride_output_m * \
        offs_m[:, None] + stride_output_n * offs_n[None, :]

    if EVEN_N:
        tl.store(c_ptrs, acc)
    else:
        mask_n = offs_n[None, :] < N
        tl.store(c_ptrs, acc, mask=mask_n)


@triton.jit
def _matmul(
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
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr
):
    offs_m = start_off + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = other + \
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

    acc = acc.to(output.dtype.element_ty)
    c_ptrs = output + stride_output_m * \
        offs_m[:, None] + stride_output_n * offs_n[None, :]
    if EVEN_N:
        if MASK_M:
            tl.store(c_ptrs, acc, mask=mask_m)
        else:
            tl.store(c_ptrs, acc)
    else:
        mask_n = offs_n[None, :] < N
        if MASK_M:
            tl.store(c_ptrs, acc, mask=mask_m & mask_n)
        else:
            tl.store(c_ptrs, acc, mask_n)


@triton.jit
def _fused_matmul(
    pid_n, start_off, end_off,
    input, other, output,
    K, N,
    stride_input_m, stride_input_k,
    stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    offs_m = start_off + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)
    rn = tl.max_contiguous(tl.multiple_of(offs_n % N, TILE_N), TILE_N)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = other + \
        (offs_k[:, None] * stride_other_k + rn[None, :] * stride_other_n)

    acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
    mask_m = offs_m[:, None] < end_off
    mask_n = offs_n[None, :] < N

    k_iters = (K // TILE_K if EVEN_K else tl.cdiv(K, TILE_K)) * BLOCK_SIZE
    k = 0
    for _ in range(0, k_iters):
        a = tl.load(input_ptrs, mask=mask_m & (offs_k[None, :] + k * TILE_K < K), other=0.0)
        b = tl.load(other_ptrs, mask=(offs_k[:, None] + k * TILE_K < K), other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        c_ptrs = output + stride_output_m * offs_m[:, None] + stride_output_n * offs_n[None, :]
        if k % BLOCK_SIZE == BLOCK_SIZE - 1:
            tl.store(c_ptrs, acc.to(output.dtype.element_ty), mask_n & mask_m)
            acc = acc - acc
            k = 0
            offs_m += TILE_M
            mask_m = offs_m[:, None] < end_off
            input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
            other_ptrs = other + \
                (offs_k[:, None] * stride_other_k + rn[None, :] * stride_other_n)
        else:
            input_ptrs += TILE_K * stride_input_k
            other_ptrs += TILE_K * stride_other_k
            k += 1


@triton.jit
def _fast_matmul_core(
    start_off_m, start_off_n,
    input, other, output,
    stride_input_m, stride_input_k,
    stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    K_ITER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr
):
    offs_m = start_off_m + tl.arange(0, TILE_M)
    offs_n = start_off_n + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = other + \
        (offs_k[:, None] * stride_other_k + offs_n[None, :] * stride_other_n)

    acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)

    for _ in range(0, K_ITER):
        a = tl.load(input_ptrs)
        b = tl.load(other_ptrs)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += TILE_K * stride_input_k
        other_ptrs += TILE_K * stride_other_k

    acc = acc.to(output.dtype.element_ty)
    c_ptrs = output + stride_output_m * \
        offs_m[:, None] + stride_output_n * offs_n[None, :]
    tl.store(c_ptrs, acc)


@triton.jit
def _fast_matmul_inline(
    start_off_m, start_off_n,
    input, other, output,
    stride_input_m, stride_input_k,
    stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    K_ITER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr
):
    _fast_matmul_core(
        start_off_m, start_off_n,
        input, other, output,
        stride_input_m, stride_input_k,
        stride_other_k, stride_other_n,
        stride_output_m, stride_output_n,
        out_dtype=out_dtype,
        K_ITER=K_ITER,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K
    )


@triton.jit(noinline=True)
def _fast_matmul_noinline(
    start_off_m, start_off_n,
    input, other, output,
    stride_input_m, stride_input_k,
    stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    out_dtype: tl.constexpr,
    K_ITER: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr
):
    _fast_matmul_core(
        start_off_m, start_off_n,
        input, other, output,
        stride_input_m, stride_input_k,
        stride_other_k, stride_other_n,
        stride_output_m, stride_output_n,
        out_dtype=out_dtype,
        K_ITER=K_ITER,
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K
    )


@triton.jit
def _dynamic_k_matmul(
    pid_k, pid_n, type_id,
    input, grad_output, grad_other,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    K, N, M,
    out_dtype: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
):
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_m = tl.arange(0, TILE_M)
    acc = tl.zeros((TILE_K, TILE_N), dtype=out_dtype)
    mask_k = offs_k[:, None] < K
    mask_n = offs_n[None, :] < N

    # [M, K] -> [K, M]
    input_ptrs = input + (offs_m[None, :] * stride_input_m + offs_k[:, None] * stride_input_k)
    # [M, N]
    grad_output_ptrs = grad_output + (offs_m[:, None] * stride_grad_output_m + offs_n[None, :] * stride_grad_output_n)

    for m in range(0, tl.cdiv(M, TILE_M)):
        a = tl.load(input_ptrs, mask=mask_k & (offs_m[None, :] + m * TILE_M < M), other=0.0)
        b = tl.load(grad_output_ptrs, mask=mask_n & (offs_m[:, None] + m * TILE_M < M), other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += TILE_M * stride_input_m
        grad_output_ptrs += TILE_M * stride_grad_output_m

    acc = acc.to(grad_other.dtype.element_ty)
    c_ptrs = grad_other + \
        stride_grad_other_k * offs_k[:, None] + stride_grad_other_n * offs_n[None, :]
    c_mask = mask_k & mask_n
    tl.store(c_ptrs, acc, mask=c_mask)
