import torch
import triton
from torch_geometric.nn import RGCNConv

from fasten import TensorSlice, compact_tensor_types
from fasten.nn import FastenRGCNConv

device = torch.device("cuda")


def slices_to_tensor(tensor_slice: TensorSlice):
    tensor = torch.zeros((tensor_slice.data_size,), dtype=torch.long, device=device)
    for i in range(len(tensor_slice)):
        s = tensor_slice.get_slice_from_index(i, is_tensor=False)
        t = tensor_slice.get_type_from_index(i, is_tensor=False)
        tensor[s] = t
    return tensor


def test_correctness():
    torch.manual_seed(12345)
    x = torch.randn(4, 4).to(device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
    ]).to(device)
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3]).to(device)
    tensor_slice = compact_tensor_types(types=edge_type, data=edge_index, dim=1, device=device)
    sorted_edge_type = slices_to_tensor(tensor_slice)

    fasten_rgcn_conv = FastenRGCNConv(4, 32, 8, is_sorted=True).to(device)
    rgcn_conv_out = fasten_rgcn_conv(x, edge_index=tensor_slice.data, edge_type=sorted_edge_type)
    fasten_rgcn_conv_out = fasten_rgcn_conv(x, edge_index=tensor_slice.data, edge_type=None, edge_tensor_slice=tensor_slice)

    assert fasten_rgcn_conv_out.shape == rgcn_conv_out.shape
    torch.testing.assert_close(fasten_rgcn_conv_out, rgcn_conv_out)


def test_benchmark():
    torch.manual_seed(12345)
    num_nodes = 10000
    num_features = 32
    num_edges = 2000
    num_types = 8

    x = torch.randn(num_nodes, num_features, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_type = torch.randint(0, num_types, (num_edges,), device=device)

    tensor_slice = compact_tensor_types(types=edge_type, data=edge_index, dim=1, device=device)
    sorted_edge_type = slices_to_tensor(tensor_slice)
    rgcn_conv = RGCNConv(num_features, num_features, num_types, is_sorted=True).to(device)
    ms = triton.testing.do_bench(lambda: rgcn_conv(x, tensor_slice.data, sorted_edge_type))
    print("Torch time:", ms)

    fasten_rgcn_conv = FastenRGCNConv(num_features, num_features, num_types, is_sorted=True).to(device)
    ms = triton.testing.do_bench(lambda: fasten_rgcn_conv(x, tensor_slice.data, None, tensor_slice))
    print("Fasten time:", ms)
