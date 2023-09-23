from typing import Tuple

import torch
import triton
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import index_sort

from fasten import TensorSlice, compact_tensor_types
from fasten.nn import FastenRGCNConv

device = torch.device("cuda")


def tensor_slice_gen(edge_type, edge_index, num_relations) -> Tuple[TensorSlice, torch.Tensor, torch.Tensor]:

    if (edge_type[1:] < edge_type[:-1]).any():
        edge_type, perm = index_sort(
            edge_type, max_value=num_relations)
        edge_index = edge_index[:, perm]
    tensor_slice = compact_tensor_types(types=edge_type, is_sorted=True, device=device)
    return tensor_slice, edge_index, edge_type


def test_correctness():

    x = torch.randn(4, 4).to(device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
    ]).to(device)
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3]).to(device)
    num_types = 8
    torch.manual_seed(12345)

    rgcn_conv = RGCNConv(4, 32, 8).to(device)
    rgcn_conv_out = rgcn_conv(x, edge_index, edge_type)

    tensor_slice, edge_index, edge_type = tensor_slice_gen(edge_type, edge_index, num_types)
    torch.manual_seed(12345)
    fasten_rgcn_conv = FastenRGCNConv(4, 32, 8).to(device)
    fasten_rgcn_conv_out = fasten_rgcn_conv(x, edge_index, edge_type, tensor_slice)

    assert fasten_rgcn_conv_out.shape == rgcn_conv_out.shape
    torch.testing.assert_close(fasten_rgcn_conv_out, rgcn_conv_out, atol=1e-3, rtol=1e-2)


def test_benchmark():

    num_nodes = 10000
    num_features = 32
    num_edges = 2000
    num_types = 8

    x = torch.randn(num_nodes, num_features, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_type = torch.randint(0, num_types, (num_edges,), device=device)

    tensor_slice, edge_index, edge_type = tensor_slice_gen(edge_type, edge_index, num_types)
    torch.manual_seed(12345)
    rgcn_conv = RGCNConv(num_features, num_features, num_types, is_sorted=True).to(device)
    ms = triton.testing.do_bench(lambda: rgcn_conv(x, edge_index, edge_type))
    print("Torch time:", ms)

    torch.manual_seed(12345)
    fasten_rgcn_conv = FastenRGCNConv(num_features, num_features, num_types).to(device)
    ms = triton.testing.do_bench(lambda: fasten_rgcn_conv(x, edge_index, edge_type, tensor_slice))
    print("Fasten time:", ms)
