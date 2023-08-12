import torch 
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)


@triton.jit
def split_matmul_kernel(
    a_ptr, b_ptr, c_ptr, ptr_ptr,
    len_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cz, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m) 
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # -----------------------------------------------------------
    for i in range(len_ptr-1):
      start_range=tl.load(ptr_ptr+i)
      nearest_start_range=(start_range//BLOCK_SIZE_K)*BLOCK_SIZE_K
      end_range=tl.load(ptr_ptr+i+1)
      a_ptr_local = a_ptrs
      b_ptr_local = b_ptrs
      for n in range(nearest_start_range//BLOCK_SIZE_K):
        a_ptr_local += BLOCK_SIZE_K * stride_ak
        b_ptr_local += BLOCK_SIZE_K * stride_bk

      for k in range(nearest_start_range,end_range,BLOCK_SIZE_K):
        a = tl.load(a_ptr_local, mask=(offs_k[None, :] < end_range - k) & (offs_k[None, :] >= start_range - k), other=0.0)
        b = tl.load(b_ptr_local, mask=(offs_k[:, None] < end_range - k) & (offs_k[:, None] >= start_range - k), other=0.0)

        accumulator += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
      if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
      c = accumulator.to(tl.float16)
      offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
      offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      c_ptrs = c_ptr + i * stride_cz + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
      c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
      tl.store(c_ptrs, c, mask=c_mask)    
        
    #-------------------------------------------------------------

   

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


def split_matmul(inputs_t, grad_out, ptr, activation=""):
    
    M, K = inputs_t.shape
    K, N = grad_out.shape
    P = ptr.numel()-1
    len_ptr= ptr.numel()
    # Allocates output.
    c = torch.empty((P, M, N), device=inputs_t.device, dtype=inputs_t.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    split_matmul_kernel[grid](
        inputs_t, grad_out, c, ptr,
        len_ptr,
        M, N, K,
        inputs_t.stride(0), inputs_t.stride(1),
        grad_out.stride(0), grad_out.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        ACTIVATION=activation
    )
    return c

# torch.manual_seed(0)
# inputs = torch.randn(128 * 10, 128 * 10).cuda()
# ptr = torch.tensor([0, 256, 505, 1024, 128 * 10]).cuda()
# other = torch.randn(3, 128 * 10, 128 * 10).cuda()
# grad_outs=torch.randn(128 * 10, 128 * 10).cuda()
# inputs_t = inputs.transpose(-2,-1)

# triton_output= split_matmul(inputs_t, grad_outs, ptr)
# print(triton_output)
# print(triton_output.shape)