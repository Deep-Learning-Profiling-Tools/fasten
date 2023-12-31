from typing import List, Tuple

import pytest
import torch
import torch_geometric
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HEATConv, HGTConv, Linear, RGATConv, RGCNConv
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn.conv import FastenHEATConv, FastenHGTConv, FastenRGATConv, FastenRGCNConv

torch.backends.cuda.matmul.allow_tf32 = True
torch_geometric.backend.use_segment_matmul = True


def ptr_to_tensor_slice(ptr: List, data: Tensor = None, is_sorted: bool = False) -> Tuple[TensorSlice, List]:

    assert ptr is not None
    slices = [slice(ptr[i], ptr[i + 1]) for i in range(len(ptr) - 1)]
    types = torch.zeros((ptr[-1],), dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = i
    tensor_slice = compact_tensor_types(data=data, types=types, is_sorted=is_sorted, device="cuda")
    return tensor_slice, slices


def tensor_slice_gen(x_dict, edge_index_dict, meta_data, num_heads) -> Tuple[TensorSlice, Tensor, TensorSlice, List]:

    # Generating tensor_slice for HeteroDictLinear
    ptr = [0]
    for key, _ in x_dict.items():
        ptr.append(ptr[-1] + x_dict[key].shape[0])
    tensor_slice_hdl, slices = ptr_to_tensor_slice(ptr, is_sorted=True)
    slices_hdl = slices

    # Generating tensor_slice for HeteroLinear
    edge_types = meta_data[1]
    num_edge_types = len(edge_types)
    H = num_heads   # No of heads
    type_list = []
    edge_map = {edge_type: i for i, edge_type in enumerate(meta_data[1])}

    for key, _ in edge_index_dict.items():
        N = x_dict[key[0]].shape[0]
        edge_type_offset = edge_map[key]
        type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(1, N) * num_edge_types + edge_type_offset
        type_list.append(type_vec)

    type_vec = torch.cat(type_list, dim=1).flatten()
    num_types = H * len(edge_types)
    ptr = index2ptr(type_vec, num_types)
    tensor_slice_hl, _ = ptr_to_tensor_slice(ptr, is_sorted=True)

    return tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl


def heat_tensor_slice_gen(data) -> TensorSlice:

    sorted_node_type, _ = index_sort(data.node_type, len(torch.unique(data.node_type)))
    ptr = index2ptr(sorted_node_type, len(torch.unique(data.node_type)))
    tensor_slice_hl, _ = ptr_to_tensor_slice(ptr, is_sorted=True)
    return tensor_slice_hl


@pytest.mark.parametrize("device", ["cuda"])
def test_rgcn(device: str):
    torch.manual_seed(12345)
    x = torch.randn(4, 4).to(device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
    ]).to(device)
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]).to(device)
    tensor_slice = compact_tensor_types(data=None, types=edge_type, is_sorted=True, device=device)

    torch.manual_seed(12345)
    rgcn_conv = RGCNConv(4, 16, 8, aggr="add", is_sorted=True).to(device)
    torch.manual_seed(12345)
    fasten_rgcn_conv = FastenRGCNConv(4, 16, 8, aggr="add", is_sorted=True, engine=Engine.TRITON).to(device)

    rgcn_conv_out = rgcn_conv(x, edge_index, edge_type)
    fasten_rgcn_conv_out = fasten_rgcn_conv(x, edge_index, edge_type, tensor_slice)

    assert fasten_rgcn_conv_out.shape == rgcn_conv_out.shape
    torch.testing.assert_close(fasten_rgcn_conv_out, rgcn_conv_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("device", ["cuda"])
def test_hgt(device: str):

    node_types = ['x', 'y', 'w', 'z']
    x_dict = {'x': torch.randn(10, 15), 'y': torch.randn(15, 20), 'w': torch.randn(10, 10), 'z': torch.randn(25, 1)}
    edge_type = [('x', 'to', 'y'), ('y', 'to', 'x'), ('y', 'to', 'w'), ('y', 'to', 'z'), ('w', 'to', 'y'), ('z', 'to', 'y')]
    edge_index_dict = {('x', 'to', 'y'): torch.cat((torch.sort(torch.randint(10, (1, 25))).values, torch.sort(torch.randint(15, (1, 25))).values), dim=0),
                       ('y', 'to', 'x'): torch.cat((torch.sort(torch.randint(15, (1, 25))).values, torch.sort(torch.randint(10, (1, 25))).values), dim=0),
                       ('y', 'to', 'w'): torch.cat((torch.sort(torch.randint(15, (1, 30))).values, torch.sort(torch.randint(10, (1, 30))).values), dim=0),
                       ('y', 'to', 'z'): torch.cat((torch.sort(torch.randint(15, (1, 10))).values, torch.sort(torch.randint(25, (1, 10))).values), dim=0),
                       ('w', 'to', 'y'): torch.cat((torch.sort(torch.randint(10, (1, 15))).values, torch.sort(torch.randint(15, (1, 15))).values), dim=0),
                       ('z', 'to', 'y'): torch.cat((torch.sort(torch.randint(25, (1, 20))).values, torch.sort(torch.randint(15, (1, 20))).values), dim=0)
                       }

    meta_data = (node_types, edge_type)
    num_heads = 2
    tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl = tensor_slice_gen(x_dict, edge_index_dict, meta_data, num_heads)

    torch.manual_seed(12345)
    hidden_channels = 16

    lin_dict = torch.nn.ModuleDict()
    for node_type in node_types:
        lin_dict[node_type] = Linear(-1, hidden_channels)

    x_dict = {
        node_type: lin_dict[node_type](x).relu_()
        for node_type, x in x_dict.items()
    }
    data = HeteroData()
    data.x_dict = x_dict
    data.edge_index_dict = edge_index_dict
    data = data.to(device)

    torch.manual_seed(12345)
    hgt_conv = HGTConv(hidden_channels, hidden_channels, meta_data, num_heads, group='sum').to(device)

    torch.manual_seed(12345)
    fasten_hgt_conv = FastenHGTConv(hidden_channels, hidden_channels, meta_data, num_heads, group='sum', engine=Engine.TRITON).to(device)

    hgt_conv_out = hgt_conv(data.x_dict, data.edge_index_dict)
    fasten_hgt_conv_out = fasten_hgt_conv(x_dict=data.x_dict, edge_index_dict=data.edge_index_dict, tensor_slice_hl=tensor_slice_hl, type_vec=type_vec, tensor_slice_hdl=tensor_slice_hdl, slices_hdl=slices_hdl)

    assert fasten_hgt_conv_out['x'].shape == hgt_conv_out['x'].shape
    torch.testing.assert_close(fasten_hgt_conv_out['x'], hgt_conv_out['x'], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("device", ["cuda"])
def test_heat(device: str):

    hidden_channels = 16
    num_heads = 2
    x = torch.randn(60, 25).to(device)
    edge_index = torch.cat((torch.cat((torch.sort(torch.randint(10, (1, 25))).values, torch.sort(torch.randint(15, (1, 25))).values), dim=0),
                            torch.cat((torch.sort(torch.randint(15, (1, 25))).values, torch.sort(torch.randint(10, (1, 25))).values), dim=0),
                            torch.cat((torch.sort(torch.randint(15, (1, 30))).values, torch.sort(torch.randint(10, (1, 30))).values), dim=0),
                            torch.cat((torch.sort(torch.randint(15, (1, 10))).values, torch.sort(torch.randint(25, (1, 10))).values), dim=0),
                            torch.cat((torch.sort(torch.randint(10, (1, 15))).values, torch.sort(torch.randint(15, (1, 15))).values), dim=0),
                            torch.cat((torch.sort(torch.randint(25, (1, 20))).values, torch.sort(torch.randint(15, (1, 20))).values), dim=0)), dim=1)
    num_nodes = [10, 15, 10, 25]
    num_edges = [25, 25, 30, 10, 15, 20]
    node_type = [num for num, freq in enumerate(num_nodes) for _ in range(freq)]
    edge_type = [num for num, freq in enumerate(num_edges) for _ in range(freq)]
    lin_in = Linear(-1, hidden_channels).to(device)
    data = HeteroData()
    data.x = lin_in(x).relu_()
    data.edge_index = edge_index
    data.node_type = torch.tensor(node_type)
    data.edge_type = torch.tensor(edge_type)
    data.edge_attr = torch.randn((data.edge_index.shape[1], 2))
    data = data.to(device)

    tensor_slice_hl = heat_tensor_slice_gen(data)

    torch.manual_seed(12345)
    heat_conv = HEATConv(hidden_channels, hidden_channels, len(torch.unique(data.node_type)), len(torch.unique(data.edge_type)), 5, 2, 6, num_heads, concat=False).to(device)
    torch.manual_seed(12345)
    fasten_heat_conv = FastenHEATConv(hidden_channels, hidden_channels, len(torch.unique(data.node_type)), len(torch.unique(data.edge_type)), 5, 2, 6,
                                      num_heads, concat=False, engine=Engine.TRITON).to(device)

    heat_conv_out = heat_conv(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr)
    fasten_heat_conv_out = fasten_heat_conv(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr, tensor_slice_hl=tensor_slice_hl)

    assert fasten_heat_conv_out.shape == heat_conv_out.shape
    torch.testing.assert_close(fasten_heat_conv_out, heat_conv_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("device", ["cuda"])
def test_rgat(device: str):

    torch.manual_seed(12345)
    x = torch.randn(4, 16).to(device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
    ]).to(device)
    edge_type = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]).to(device)
    tensor_slice = compact_tensor_types(data=None, types=edge_type, is_sorted=True, device=device)
    assert tensor_slice is not None

    torch.manual_seed(12345)
    rgat_conv = RGATConv(16, 16, 8).to(device)
    torch.manual_seed(12345)
    fasten_rgat_conv = FastenRGATConv(16, 16, 8, engine=Engine.TRITON).to(device)

    rgat_conv_out = rgat_conv(x, edge_index, edge_type)
    fasten_rgat_conv_out = fasten_rgat_conv(x, edge_index, edge_type, tensor_slice=tensor_slice)

    assert fasten_rgat_conv_out.shape == rgat_conv_out.shape
    torch.testing.assert_close(fasten_rgat_conv_out, rgat_conv_out, rtol=1e-2, atol=1e-2)
