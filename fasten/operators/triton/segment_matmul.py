import torch
import triton
import triton.language as tl
import os
import numpy as np
import bisect
import pyg_lib

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)



@triton.jit
def segment_matmul_kernel(
    inputs_ptr, others_ptr, c_ptr,
    M, N, K,
    stride_input_m, stride_input_k,
    stride_other_z, stride_other_k,stride_other_n,
    stride_output_m, stride_output_n,
    start_bounds, end_bounds, batch_idx, len_m,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr= 8,
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

   
    start_offs_input_m= tl.load(start_bounds+ pid_m)
    end_offs_input_m= tl.load(end_bounds+ pid_m)
    batch_idx_other= tl.load(batch_idx + pid_m)
    if start_offs_input_m >= end_offs_input_m:
      return

    offs_input_m=(start_offs_input_m + tl.arange(0, BLOCK_SIZE_M)) 
    offs_other_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)


    input_ptrs = inputs_ptr + \
         (offs_input_m[:, None] * stride_input_m + offs_k[None, :] * stride_input_k)
    other_ptrs = others_ptr + batch_idx_other * stride_other_z + \
         (offs_k[:, None] * stride_other_k + offs_other_n[None, :] * stride_other_n)
 


    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    input_mask=offs_input_m[:, None]<end_offs_input_m

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):

        input_k_mask = (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(input_ptrs, mask=input_mask & input_k_mask, other=0.0)
        b = tl.load(other_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        input_ptrs += BLOCK_SIZE_K * stride_input_k
        other_ptrs += BLOCK_SIZE_K * stride_other_k

    c = accumulator.to(tl.float32)

    offs_cm= (start_offs_input_m + tl.arange(0, BLOCK_SIZE_M)) 
    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) 
    c_ptrs = c_ptr + stride_output_m * offs_cm[:, None] + stride_output_n * offs_cn[None, :]
    c_mask = (offs_cm[:,None]<end_offs_input_m ) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)





def triton_segment_matmul(inputs, ptr, other, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
  M, K = inputs.shape
  P, K, N = other.shape
  start_bounds=[]
  for i in range(len(ptr)-1):
    if ptr[i]== ptr[i+1]:
      for j in range(ptr[i], ptr[i+1]+1, BLOCK_M):
        start_bounds.append(j)
    else:
      for j in range(ptr[i], ptr[i+1], BLOCK_M):
        start_bounds.append(j)

  end_bounds= start_bounds[1:]
  end_bounds.append(ptr[-1])
  end_bounds=torch.tensor(end_bounds, dtype=torch.int32)
  
  len_m= len(start_bounds)
  batch_idx=[]
  len_ptr=1
  matrix_count=0
  for j in range(0,len_m):
      if start_bounds[j]//ptr[len_ptr]==0:
        batch_idx.append(matrix_count)
      else:
        matrix_count+=1
        batch_idx.append(matrix_count)
        len_ptr+=1
  
  start_bounds=torch.tensor(start_bounds, dtype=torch.int32)
  assert len(start_bounds)==len(batch_idx)
  batch_idx=torch.tensor(batch_idx, dtype=torch.int32)


  start_bounds=start_bounds.cuda()
  end_bounds= end_bounds.cuda()
  batch_idx= batch_idx.cuda()




  output = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype, requires_grad= True)
   
  grid = lambda META: (
         len_m * triton.cdiv(N, BLOCK_N),
  )
  start_event = torch.cuda.Event(enable_timing = True)
  end_event = torch.cuda.Event(enable_timing = True)
  start_event.record()
  segment_matmul_kernel[grid](
        inputs, other, output,
        M, N, K,
        inputs.stride(0), inputs.stride(1),
        other.stride(0), other.stride(1), other.stride(2),
        output.stride(0), output.stride(1),
        start_bounds, end_bounds, batch_idx, len_m,
  )
  end_event.record()
  torch.cuda.synchronize()
  print("Triton time:",start_event.elapsed_time(end_event))
  return output


torch.manual_seed(42)
inputs = torch.randn(1029, 16).cuda()
ptr=torch.tensor([0,257,270,377,377,1029,1029, 1029]).cuda()
other = torch.randn(7, 16, 4).cuda()
triton_output= triton_segment_matmul(inputs, ptr, other)
torch_output = pyg_lib.ops.segment_matmul(inputs, ptr, other)
print(torch.max(torch.abs(triton_output-torch_output)))
