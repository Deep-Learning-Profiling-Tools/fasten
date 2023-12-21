import argparse
import os.path as osp
from typing import Tuple

import torch
import torch.nn.functional as F
import torch_geometric
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.datasets import Entities
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import index_sort, k_hop_subgraph
from triton.testing import do_bench

from fasten import TensorSlice, compact_tensor_types
from fasten.nn import FastenRGCNConv

torch.backends.cuda.matmul.allow_tf32 = True
torch_geometric.backend.use_segment_matmul = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
parser.add_argument('--mode', type=str, default='pyg',
                    choices=['pyg', 'fasten'])
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
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

data.num_nodes = node_idx.size(0)
data.edge_index = edge_index
data.edge_type = data.edge_type[edge_mask]
data.train_idx = mapping[:data.train_idx.size(0)]
data.test_idx = mapping[data.train_idx.size(0):]


def tensor_slice_gen(edge_type, edge_index, num_relations) -> Tuple[TensorSlice, torch.Tensor, torch.Tensor]:
    if (edge_type[1:] < edge_type[:-1]).any():
        edge_type, perm = index_sort(
            edge_type, max_value=num_relations)
        edge_index = edge_index[:, perm]
    tensor_slice = compact_tensor_types(data=None, types=edge_type, is_sorted=True, device=device)
    return tensor_slice, edge_index, edge_type


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(data.num_nodes, args.hidden_size, dataset.num_relations, aggr="add", is_sorted=True)
        self.conv2 = RGCNConv(args.hidden_size, dataset.num_classes, dataset.num_relations, aggr="add", is_sorted=True)

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


class FastenNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FastenRGCNConv(data.num_nodes, args.hidden_size, dataset.num_relations, aggr="add", is_sorted=True)
        self.conv2 = FastenRGCNConv(args.hidden_size, dataset.num_classes, dataset.num_relations, aggr="add", is_sorted=True)

    def forward(self, edge_index, edge_type, tensor_slice):
        x = F.relu(self.conv1(None, edge_index, edge_type, tensor_slice))
        x = self.conv2(x, edge_index, edge_type, tensor_slice)
        return F.log_softmax(x, dim=1)


if args.mode == "fasten":
    model, data = FastenNet().to(device), data.to(device)
else:
    model, data = Net().to(device), data.to(device)
tensor_slice, edge_index, edge_type = tensor_slice_gen(data.edge_type, data.edge_index, dataset.num_relations)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    with record_function("RGCN Train"):
        optimizer.zero_grad()
        model.train()
        with record_function("RGCN Inference"):
            if args.mode == "fasten":
                out = model(edge_index, edge_type, tensor_slice)
            else:
                out = model(edge_index, edge_type)
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        loss.backward()
        optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    if args.mode == "fasten":
        pred = model(edge_index, edge_type, tensor_slice).argmax(dim=-1)
    else:
        pred = model(edge_index, edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc


if args.profile == "none":
    for epoch in range(1, 5):
        loss = train()
        train_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
              f'Test: {test_acc:.4f}')

elif args.profile == "profile":
    # warmup
    train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
        for epoch in range(1, 5):
            train()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

else:  # args.profile == "benchmark"
    def pyg_fn():
        model(edge_index, edge_type).argmax(dim=-1)

    def fasten_fn():
        model(edge_index, edge_type, tensor_slice).argmax(dim=-1)
    fn = pyg_fn if args.mode == "pyg" else fasten_fn
    inference_ms = do_bench(fn)
    train_ms = do_bench(train)
    print(f"{args.mode} inference: {inference_ms} ms")
    print(f"{args.mode} train: {train_ms} ms")
