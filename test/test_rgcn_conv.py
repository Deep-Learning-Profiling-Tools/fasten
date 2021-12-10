import torch

from torch_geometric.nn import RGCNConv, FastRGCNConv
from fasten.nn import FastenRGCNConv
from fasten import Ops as ops


def correctness_float(mode='add'):
    x = torch.randn(4, 4)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
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


def correctness_long(mode='add'):
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [0, 0, 1, 2, 1, 1, 0, 0, 1, 0, 1, 3],
    ])
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3])

    torch.manual_seed(12345)
    rgcn_conv = RGCNConv(4, 32, 8, num_bases=4, aggr=mode)
    rgcn_conv_out = rgcn_conv(
        x=None, edge_index=edge_index, edge_type=edge_type)

    torch.manual_seed(12345)
    edge_index, edge_type = ops.compact(
        edge_index, edge_type, type_dim=1)
    fasten_rgcn_conv = FastenRGCNConv(4, 32, 8, num_bases=4, aggr=mode)
    fasten_rgcn_conv_out = fasten_rgcn_conv(
        x=None, edge_index=edge_index, edge_type=edge_type)

    assert(fasten_rgcn_conv_out.shape == rgcn_conv_out.shape)
    assert(torch.allclose(fasten_rgcn_conv_out, rgcn_conv_out) is True)


def test_correctness():
    correctness_float('add')
    correctness_float('mean')
    correctness_long('add')
    correctness_long('mean')


def speedup_gpu():
    num_nodes = 1000
    num_features = 32
    num_edges = 20000
    num_types = 32
    num_bases = 4

    device = torch.device('cuda:0')
    x = torch.randn(num_nodes, num_features, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_type = torch.randint(0, num_types, (num_edges,), device=device)

    def run(test_name, edge_index, edge_type):
        repeat = 3
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.manual_seed(12345)
        if test_name == 'rgcn_conv':
            rgcn_conv = RGCNConv(num_features, num_features,
                                 num_types, num_bases=num_bases, aggr='mean').to(device)
        elif test_name == 'fast_rgcn_conv':
            rgcn_conv = FastRGCNConv(
                num_features, num_features, num_types, num_bases=num_bases, aggr='mean').to(device)
        elif test_name == 'fasten_rgcn_conv':
            edge_index, edge_type = ops.compact(
                edge_index, edge_type, type_dim=1)
            rgcn_conv = FastenRGCNConv(
                num_features, num_features, num_types, num_bases=num_bases, aggr='mean').to(device)

        # warmup
        rgcn_conv_out = rgcn_conv(x, edge_index, edge_type)

        # forward
        start_event.record()
        for _ in range(repeat):
            rgcn_conv_out = rgcn_conv(x, edge_index, edge_type)
        end_event.record()
        end_event.synchronize()

        ms = start_event.elapsed_time(end_event) / repeat

        print('{} forward: {} ms'.format(test_name, ms))

        return rgcn_conv_out

    rgcn_conv_out = run('rgcn_conv', edge_index, edge_type)
    fast_rgcn_conv_out = run('fast_rgcn_conv', edge_index, edge_type)
    fasten_rgcn_conv_out = run('fasten_rgcn_conv', edge_index, edge_type)

    assert(torch.allclose(rgcn_conv_out, fast_rgcn_conv_out, atol=1e-3) is True)
    assert(torch.allclose(fasten_rgcn_conv_out,
           fast_rgcn_conv_out, atol=1e-3) is True)
    assert(torch.allclose(fasten_rgcn_conv_out, rgcn_conv_out, atol=1e-3) is True)


def test_speedup():
    speedup_gpu()
