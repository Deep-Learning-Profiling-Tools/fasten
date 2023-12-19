import os.path as osp
import time
import argparse

import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor

from torch_geometric.datasets import Entities
from torch_geometric.nn import RGATConv
from torch_geometric.utils import k_hop_subgraph

from fasten.nn import FastenRGATConv
from fasten import Engine, TensorSlice, compact_tensor_types


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
parser.add_argument('--mode', type=str, default='pyg',
                    choices=['pyg', 'fasten'])
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()

device = torch.device(args.device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0]
node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx, 2, data.edge_index, relabel_nodes=True)
data.x = torch.randn(data.num_nodes, 16)

def ptr_to_tensor_slice(ptr: List, data: Tensor = None, is_sorted: bool = False) -> TensorSlice:

    assert ptr is not None
    slices = [slice(ptr[i], ptr[i + 1]) for i in range(len(ptr) - 1)]
    types = torch.zeros((ptr[-1],), dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = i
    tensor_slice = compact_tensor_types(data=data, types=types, is_sorted=is_sorted, device=device)
    return tensor_slice


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
    model = FastenRGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)
    ptr = [i for i in range(len(data.edge_type)+1)]
    tensor_slice = ptr_to_tensor_slice(ptr, is_sorted=True)
    assert tensor_slice is not None
else:
    model = RGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    if args.mode == "fasten":
        out = model(data.x, data.edge_index, data.edge_type, tensor_slice=tensor_slice)
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
        pred = model(data.x, data.edge_index, data.edge_type, tensor_slice=tensor_slice).argmax(dim=-1)
    else:
        pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc


times = []
for epoch in range(1, 5):
    start = time.time()
    loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

