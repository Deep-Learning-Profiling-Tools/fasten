import torch

from torch_geometric.nn import RGCNConv, FastRGCNConv
from fasten.nn import FastenRGCNConv
from fasten import Ops as ops


def correctness_float(mode='add'):
    x = torch.randn(4, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [0, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 3],
    ])
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3])

    torch.manual_seed(12345)
    rgcn_conv = RGCNConv(4, 32, 8, num_bases=4, aggr=mode)
    rgcn_conv_out = rgcn_conv(x, edge_index, edge_type)

    torch.manual_seed(12345)
    edge_index, edge_type = ops.compact(
        edge_index, edge_type, type_dim=1)
    fasten_rgcn_conv = FastenRGCNConv(4, 32, 8, num_bases=4, aggr=mode)
    fasten_rgcn_conv_out = fasten_rgcn_conv(x, edge_index, edge_type)

    assert(fasten_rgcn_conv_out.shape == rgcn_conv_out.shape)
    assert(torch.allclose(fasten_rgcn_conv_out, rgcn_conv_out) is True)


def correctness_long():
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [0, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 3],
    ])
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3])

    torch.manual_seed(12345)
    rgcn_conv = RGCNConv(4, 32, 8, num_bases=4)
    rgcn_conv_out = rgcn_conv(
        x=None, edge_index=edge_index, edge_type=edge_type)

    torch.manual_seed(12345)
    edge_index, edge_type = ops.compact(
        edge_index, edge_type, type_dim=1)
    fasten_rgcn_conv = FastenRGCNConv(4, 32, 8, num_bases=4)
    fasten_rgcn_conv_out = fasten_rgcn_conv(
        x=None, edge_index=edge_index, edge_type=edge_type)

    assert(fasten_rgcn_conv_out.shape == rgcn_conv_out.shape)
    assert(torch.allclose(fasten_rgcn_conv_out, rgcn_conv_out) is True)


def test_correctness():
    correctness_float('add')
    # correctness_long()


def test_speedup():
    pass
