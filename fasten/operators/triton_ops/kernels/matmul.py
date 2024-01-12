import triton
import triton.language as tl


@triton.jit
def _reg_matmul(
    pid_n, type_id,
    start_off,
    input, other, output, N,
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
    other_ptrs = other + type_id * stride_other_b + \
        (offs_k[:, None] * stride_other_k + rn[None, :] * stride_other_n)
    b = tl.load(other_ptrs)

    # [M, K] x [K, N] -> [M, N]
    input_ptrs = input + (offs_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    output_ptrs = output + stride_output_m * offs_m[:, None] + stride_output_n * offs_n[None, :]
    for _ in range(0, BLOCK_SIZE):
        a = tl.load(input_ptrs)
        acc = tl.dot(a, b, out_dtype=out_dtype).to(output.dtype.element_ty)
        if EVEN_N:
            tl.store(output_ptrs, acc)
        else:
            mask_n = offs_n[None, :] < N
            tl.store(output_ptrs, acc, mask=mask_n)
        input_ptrs += TILE_M * stride_input_m
        output_ptrs += TILE_M * stride_output_m


@triton.jit
def _general_matmul(
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
def _prefetch_matmul(
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
    output_ptrs = output + stride_output_m * offs_m[:, None] + stride_output_n * offs_n[None, :]
    original_input_ptrs = input_ptrs
    original_other_ptrs = other_ptrs

    acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
    mask_n = offs_n[None, :] < N

    k_iters = K // TILE_K if EVEN_K else tl.cdiv(K, TILE_K)
    for k in range(0, k_iters * BLOCK_SIZE):
        i = k % k_iters
        if EVEN_K:
            a = tl.load(input_ptrs)
            b = tl.load(other_ptrs)
        else:
            a = tl.load(input_ptrs, mask=offs_k[None, :] + i * TILE_K < K, other=0.0)
            b = tl.load(other_ptrs, mask=offs_k[:, None] + i * TILE_K < K, other=0.0)
        acc += tl.dot(a, b, out_dtype=out_dtype)
        if i == k_iters - 1:
            if EVEN_N:
                tl.store(output_ptrs, acc.to(output.dtype.element_ty))
            else:
                tl.store(output_ptrs, acc.to(output.dtype.element_ty), mask_n)
            output_ptrs += TILE_M * stride_output_m
        if i == k_iters - 1:
            acc = tl.zeros((TILE_M, TILE_N), dtype=out_dtype)
            original_input_ptrs += TILE_M * stride_input_m
            input_ptrs = original_input_ptrs
            other_ptrs = original_other_ptrs
        else:
            input_ptrs += TILE_K * stride_input_k
            other_ptrs += TILE_K * stride_other_k


@triton.jit
def _dynamic_matmul(
    pid_k, pid_n, next_id,
    input, grad_output, grad_other, grad_other_tiles,
    stride_input_m, stride_input_k,
    stride_grad_output_m, stride_grad_output_n,
    stride_grad_other_b, stride_grad_other_k, stride_grad_other_n,
    K, N, M, length,
    out_dtype: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    DETERMINISTIC: tl.constexpr
):
    offs_k = pid_k * TILE_K + tl.arange(0, TILE_K)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_m = tl.arange(0, TILE_M)
    acc = tl.zeros((TILE_K, TILE_N), dtype=out_dtype)
    mask_k = offs_k[:, None] < K if not EVEN_K else True
    mask_n = offs_n[None, :] < N if not EVEN_N else True

    # [M, K] -> [K, M]
    input_ptrs = input + (offs_m[None, :] * stride_input_m + offs_k[:, None] * stride_input_k)
    # [M, N]
    grad_output_ptrs = grad_output + (offs_m[:, None] * stride_grad_output_m + offs_n[None, :] * stride_grad_output_n)

    m_iter = length // TILE_M if EVEN_M else tl.cdiv(length, TILE_M)
    for m in range(0, m_iter):
        if EVEN_K:
            if EVEN_M:
                a = tl.load(input_ptrs)
            else:
                a = tl.load(input_ptrs, mask=(offs_m[None, :] + m * TILE_M < length), other=0.0)
        else:
            if EVEN_M:
                a = tl.load(input_ptrs, mask=mask_k, other=0.0)
            else:
                a = tl.load(input_ptrs, mask=mask_k & (offs_m[None, :] + m * TILE_M < length), other=0.0)
        if EVEN_N:
            if EVEN_M:
                b = tl.load(grad_output_ptrs)
            else:
                b = tl.load(grad_output_ptrs, mask=(offs_m[:, None] + m * TILE_M < length), other=0.0)
        else:
            if EVEN_M:
                b = tl.load(grad_output_ptrs, mask=mask_n)
            else:
                b = tl.load(grad_output_ptrs, mask=mask_n & (offs_m[:, None] + m * TILE_M < length), other=0.0)

        acc += tl.dot(a, b, out_dtype=out_dtype)
        input_ptrs += TILE_M * stride_input_m
        grad_output_ptrs += TILE_M * stride_grad_output_m

    acc = acc.to(grad_other.dtype.element_ty)

    if DETERMINISTIC:
        c_ptrs = grad_other_tiles + \
            next_id * stride_grad_other_k * offs_k[:, None] + stride_grad_other_n * offs_n[None, :]
        if EVEN_N and EVEN_K:
            tl.store(c_ptrs, acc)
        else:
            c_mask = mask_k & mask_n
            tl.store(c_ptrs, acc, mask=c_mask)
    else:
        c_ptrs = grad_other + \
            stride_grad_other_k * offs_k[:, None] + stride_grad_other_n * offs_n[None, :]
        if M <= TILE_M * m_iter:
            if EVEN_N and EVEN_K:
                tl.store(c_ptrs, acc)
            else:
                c_mask = mask_k & mask_n
                tl.store(c_ptrs, acc, mask=c_mask)
        else:
            if EVEN_N and EVEN_K:
                tl.atomic_add(c_ptrs, acc)
            else:
                c_mask = mask_k & mask_n
                tl.atomic_add(c_ptrs, acc, mask=c_mask)
