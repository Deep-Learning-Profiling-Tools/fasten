import argparse
import os.path as osp
import time
from typing import Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch.profiler import ProfilerActivity, profile
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGATConv
from torch_geometric.utils import index_sort, k_hop_subgraph
from triton.testing import do_bench

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn import FastenRGATConv

torch.backends.cuda.matmul.allow_tf32 = True
torch_geometric.backend.use_segment_matmul = True

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
parser.add_argument('--mode', type=str, default='pyg',
                    choices=['pyg', 'fasten'])
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
parser.add_argument('--profile', type=str, default='none',
                    choices=['none', 'profile', 'benchmark'])
parser.add_argument('--hidden_size', type=int, default=32)
args = parser.parse_args()

device = torch.device(args.device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0]
node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx, 2, data.edge_index, relabel_nodes=True)
data.x = torch.randn(data.num_nodes, args.hidden_size)


def tensor_slice_gen(edge_type, edge_index, num_relations) -> Tuple[TensorSlice, torch.Tensor, torch.Tensor]:
    if (edge_type[1:] < edge_type[:-1]).any():
        edge_type, perm = index_sort(
            edge_type, max_value=num_relations)
        edge_index = edge_index[:, perm]
    tensor_slice = compact_tensor_types(data=None, types=edge_type, is_sorted=True, device=device)
    return tensor_slice, edge_index, edge_type


class FastenRGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations):
        super().__init__()
        self.conv1 = FastenRGATConv(in_channels, hidden_channels, num_relations, engine=Engine.TRITON)
        self.conv2 = FastenRGATConv(hidden_channels, hidden_channels, num_relations, engine=Engine.TRITON)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type, tensor_slice):
        x = self.conv1(x, edge_index, edge_type, tensor_slice=tensor_slice).relu()
        x = self.conv2(x, edge_index, edge_type, tensor_slice=tensor_slice).relu()
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_relations):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


data = data.to(device)
if args.mode == "fasten":
    model = FastenRGAT(args.hidden_size, args.hidden_size, dataset.num_classes, dataset.num_relations).to(device)
    ptr = [i for i in range(len(data.edge_type) + 1)]
    tensor_slice, edge_index, edge_type = tensor_slice_gen(data.edge_type, data.edge_index, dataset.num_relations)
    assert tensor_slice is not None
else:
    model = RGAT(args.hidden_size, args.hidden_size, dataset.num_classes, dataset.num_relations).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    if args.mode == "fasten":
        out = model(data.x, edge_index, edge_type, tensor_slice=tensor_slice)
    else:
        out = model(data.x, data.edge_index, data.edge_type)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    if args.mode == "fasten":
        pred = model(data.x, edge_index, edge_type, tensor_slice=tensor_slice).argmax(dim=-1)
    else:
        pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc


if args.profile == "none":
    times = []
    for epoch in range(1, 5):
        start = time.time()
        loss = train()
        train_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
              f'Test: {test_acc:.4f}')
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

elif args.profile == "profile":
    # warmup
    train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
        for epoch in range(1, 5):
            train()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

else:  # args.profile == "benchmark"
    def train_fn():
        train()
    train_ms = do_bench(train_fn)
    print(f"{args.mode} train: {train_ms} ms")
