import torch
from fasten import Engine, compact_tensor_types, ops


slices = [slice(0, 127), slice(127, 256), slice(256, 257), slice(257, 512)]

types = torch.zeros(512, dtype=torch.long, device="cuda", requires_grad=False)
for i, s in enumerate(slices):
    types[s]= i
    print(i)
    print(s)
print("Types:", types)
data = torch.randn((512, 2), device="cuda")
print("Data:",data)
sorted_data, tensor_slice = compact_tensor_types(data, types, device="cuda")
print("Sorted data:", sorted_data)
print("Tensor slice:",tensor_slice)
# output = ops.fasten_segment_matmul(sorted_data, tensor_slice.slices, other, engine, tensor_slice)