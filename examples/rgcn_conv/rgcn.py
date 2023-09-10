import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Entities
from torch_geometric.nn import Linear, RGCNConv
from torch_geometric.utils import index_sort, k_hop_subgraph
from torch_geometric.utils.sparse import index2ptr

from fasten import Engine, TensorSlice, compact_tensor_types, ops
from fasten.nn import FastenRGCNConv

torch.manual_seed(12345)

device = torch.device('cuda')

def data_prep_rgcn(dataset_arg):
    path =osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
    dataset = Entities(path, dataset_arg)
    data = dataset[0]
    node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
    node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx, 2, data.edge_index, relabel_nodes=True)

    data.num_nodes = node_idx.size(0)
    data.edge_index = edge_index
    data.edge_type = data.edge_type[edge_mask]
    data.train_idx = mapping[:data.train_idx.size(0)]
    data.test_idx = mapping[data.train_idx.size(0):]
    return data, dataset.num_relations, dataset.num_classes

def tensor_slice_gen(data, num_relations):
    edge_type = data.edge_type
    edge_index= data.edge_index
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


def test_rgcn(data, num_relations, num_classes):
    torch.cuda.empty_cache()
    tensor_slice = tensor_slice_gen(data, num_relations)

    data = data.to(device)
    rgcn_conv1 = RGCNConv(data.num_nodes, 32, num_relations).to(device)
    rgcn_conv2= RGCNConv(32, num_classes, num_relations).to(device)

    x = torch.randn((data.num_nodes, 32), device="cuda")
    rgcn_conv_out = rgcn_conv2(x, data.edge_index, data.edge_type)

    fasten_rgcn_conv1 = FastenRGCNConv(data.num_nodes, 32, num_relations).to(device)
    fasten_rgcn_conv2 = FastenRGCNConv(32, num_classes, num_relations).to(device)
    fasten_rgcn_conv_out = fasten_rgcn_conv2(x= x, edge_index= data.edge_index, edge_type= data.edge_type, edge_tensor_slice=tensor_slice)

    assert(fasten_rgcn_conv_out.shape == rgcn_conv_out.shape)
    print("Max Difference:",torch.max(torch.abs(fasten_rgcn_conv_out- rgcn_conv_out)))


def rgcn_test_loop():
    datasets= ['AIFB', 'MUTAG', 'BGS', 'AM']
    for data_args in datasets:
        print("Testing on:", data_args)
        data, num_relations, num_classes = data_prep_rgcn(data_args)
        test_rgcn(data, num_relations, num_classes)

rgcn_test_loop()
