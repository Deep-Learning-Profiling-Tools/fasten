import torch
import triton
import triton.language as tl
import pyg_lib


@triton.jit
def segment_matmul_kernel(
    inputs_ptr, others_ptr, c_ptr,
    M, N, K,
    stride_input_m, stride_input_k,
    stride_other_b, stride_other_k, stride_other_n,
    stride_output_m, stride_output_n,
    start_bounds, end_bounds, batch_idx, len_m,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    pid = tl.program_id(axis=0)
    num_pid_m = len_m
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    start_offs_input_m = tl.load(start_bounds + pid_m)
    end_offs_input_m = tl.load(end_bounds + pid_m)

    if start_offs_input_m >= end_offs_input_m:
        return

    batch_idx_other = tl.load(batch_idx + pid_m)

    offs_input_m = start_offs_input_m + tl.arange(0, BLOCK_SIZE_M)
    offs_other_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    input_ptrs = inputs_ptr + \
        (offs_input_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = others_ptr + batch_idx_other * stride_other_b + \
        (offs_k[:, None] * stride_other_k + offs_other_n[None, :] * stride_other_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    input_mask = offs_input_m[:, None] < end_offs_input_m

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_k_mask = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(input_ptrs, mask=input_mask & input_k_mask, other=0.0)
        b = tl.load(other_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K * stride_input_k
        other_ptrs += BLOCK_SIZE_K * stride_other_k

    c = accumulator.to(tl.float32)

    offs_cm = (start_offs_input_m + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    c_ptrs = c_ptr + stride_output_m * \
        offs_cm[:, None] + stride_output_n * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < end_offs_input_m) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# TODO(Keren): this function is here just for testing purpose,
# we should have start_bounds and end_bounds on GPU when initializing the model


def triton_segment_matmul(inputs, ptr, others, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = inputs.shape
    _, K, N = others.shape

    start_bounds = []
    for i in range(len(ptr)-1):
        if ptr[i] == ptr[i+1]:
            for j in range(ptr[i], ptr[i+1]+1, BLOCK_M):
                start_bounds.append(j)
        else:
            for j in range(ptr[i], ptr[i+1], BLOCK_M):
                start_bounds.append(j)

    end_bounds = start_bounds[1:]
    end_bounds.append(ptr[-1])
    end_bounds = torch.tensor(end_bounds, dtype=torch.int32, device=inputs.device)

    num_m_blocks = len(start_bounds)
    batch_idx = []
    ptr_idx = 1
    b = 0
    for j in range(0, num_m_blocks):
        if start_bounds[j]//ptr[ptr_idx] == 0:
            batch_idx.append(b)
        else:
            b += 1
            batch_idx.append(b)
            ptr_idx += 1

    start_bounds = torch.tensor(start_bounds, dtype=torch.int32, device=inputs.device)
    batch_idx = torch.tensor(batch_idx, dtype=torch.int32, device=inputs.device)

    output = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = (num_m_blocks * triton.cdiv(N, BLOCK_N),)
    segment_matmul_kernel[grid](
        inputs, others, output,
        M, N, K,
        inputs.stride(0), inputs.stride(1),
        others.stride(0), others.stride(1), others.stride(2),
        output.stride(0), output.stride(1),
        start_bounds, end_bounds, batch_idx, num_m_blocks,
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K,
    )
    return output


def torch_segment_matmul(inputs, ptrs, others):
    M, K = inputs.shape
    _, K, N = others.shape
    output = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)

    for i in range(len(ptrs) - 1):
        if ptrs[i] == ptrs[i+1]:
            continue
        else:
            output[ptrs[i]:ptrs[i+1],
                   :] = torch.matmul(inputs[ptrs[i]:ptrs[i+1], :], others[i])
    return output


torch.manual_seed(42)
inputs = torch.randn(128*200, 16).cuda()
ptrs = torch.tensor([0, 256, 505, 505, 729, 128*10, 128*20, 128*200, 128*200]).cuda()
other = torch.randn(8, 16, 16).cuda()
triton_output = triton_segment_matmul(inputs, ptrs, other)
pyg_output = pyg_lib.ops.segment_matmul(inputs, ptrs, other)
print(torch.max(torch.abs(triton_output-pyg_output)))
