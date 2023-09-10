import torch
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn import FastenRGCNConv

device = torch.device("cuda")

def tensor_slice_gen(edge_type, edge_index, num_relations) -> TensorSlice:

    if (edge_type[1:] < edge_type[:-1]).any():
        edge_type, perm = index_sort(
            edge_type, max_value=num_relations)
        edge_index = edge_index[:, perm]
    edge_type_ptr = index2ptr(edge_type, num_relations)
    edge_ptr= [slice(edge_type_ptr[i], edge_type_ptr[i + 1]) for i in range(len(edge_type_ptr) - 1)]
    M = sum([s.stop - s.start for s in edge_ptr])
    fake_data = torch.randn((M, 1)).to(device)
    types = torch.zeros(M, dtype=torch.long, device=device, requires_grad=False)
    for i, s in enumerate(edge_ptr):
        types[s] = i
    sorted_data, tensor_slice = compact_tensor_types(fake_data, types, device=device)
    return tensor_slice


def correctness():

    x = torch.randn(4,4).to(device)
    edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
    [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
    ]).to(device)
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3]).to(device)
    num_types = 8
    torch.manual_seed(12345)

    rgcn_conv = RGCNConv(4, 32, 8).to(device)
    rgcn_conv_out = rgcn_conv(x, edge_index, edge_type)

    tensor_slice = tensor_slice_gen(edge_type, edge_index, num_types)
    fasten_rgcn_conv = FastenRGCNConv(4,32,8).to(device)
    fasten_rgcn_conv_out = fasten_rgcn_conv(x, edge_index, edge_type, tensor_slice)

    assert fasten_rgcn_conv_out.shape == rgcn_conv_out.shape
    torch.testing.assert_close(fasten_rgcn_conv_out, rgcn_conv_out, atol=1e-1, rtol=1e-2)

def benchmark_gpu():

    num_nodes = 1000
    num_features = 32
    num_edges = 20000
    num_types = 32

    x = torch.randn(num_nodes, num_features, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_type = torch.randint(0, num_types, (num_edges,), device=device)

    torch.manual_seed(12345)

    rgcn_conv = RGCNConv(num_features, num_features, num_types).to(device)

    tensor_slice = tensor_slice_gen(edge_type, edge_index, num_types)
    fasten_rgcn_conv = FastenRGCNConv(num_features, num_features, num_types).to(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    out = rgcn_conv(x, edge_index, edge_type)  # warmup
    start_event.record()
    for _ in range(100):
        rgcn_conv(x, edge_index, edge_type)
    end_event.record()
    end_event.synchronize()
    torch_time = start_event.elapsed_time(end_event) / 100

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    out = fasten_rgcn_conv(x, edge_index, edge_type, tensor_slice)  # warmup
    start_event.record()
    for _ in range(100):
        fasten_rgcn_conv(x, edge_index, edge_type, tensor_slice)
    end_event.record()
    end_event.synchronize()
    fasten_time = start_event.elapsed_time(end_event) / 100

    print("Torch time:", torch_time)
    print("Fasten time:", fasten_time)
