import argparse
import os.path as osp
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.datasets import DBLP
from torch_geometric.nn import Linear
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn import FastenHEATConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DBLP')
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
args = parser.parse_args()
device = torch.device(args.device)


# We initialize conference node features with a single one-vector as feature:
dataset = DBLP(path, transform=T.Constant(node_types='conference'))
data = dataset[0]
print(data)
author_num = data['author'].x.shape[0]
data = data.to_homogeneous()
# Create ramdom values for edge_attr
data["edge_attr"] = torch.randn((data.edge_index.shape[1], 2))
print(data)


def ptr_to_tensor_slice(ptr: List, data: Tensor = None, is_sorted: bool = False) -> Tuple[TensorSlice, List]:

    assert ptr is not None
    slices = [slice(ptr[i], ptr[i + 1]) for i in range(len(ptr) - 1)]
    types = torch.zeros((ptr[-1],), dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = i
    tensor_slice = compact_tensor_types(data=data, types=types, is_sorted=is_sorted, device=device)
    return tensor_slice, slices


def tensor_slice_gen(data) -> TensorSlice:

    sorted_node_type, _ = index_sort(data.node_type, len(torch.unique(data.node_type)))
    ptr = index2ptr(sorted_node_type, len(torch.unique(data.node_type)))
    tensor_slice_hl, _ = ptr_to_tensor_slice(ptr, is_sorted=True)
    return tensor_slice_hl


class HEAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = FastenHEATConv(hidden_channels, hidden_channels, len(torch.unique(data.node_type)), len(torch.unique(data.edge_type)), 5, 2, 6,
                                  num_heads, concat=False, engine=Engine.TRITON)
            self.convs.append(conv)

        self.lin_in = Linear(-1, hidden_channels)
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, node_type, edge_type, edge_attr, tensor_slice_hl):
        x = self.lin_in(x).relu_()

        for conv in self.convs:
            x = conv(x, edge_index, node_type, edge_type, edge_attr, tensor_slice_hl=tensor_slice_hl)

        return self.lin_out(x)


model = HEAT(hidden_channels=64, out_channels=4, num_heads=2, num_layers=1)
data, model = data.to(device), model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
tensor_slice_hl = tensor_slice_gen(data)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr, tensor_slice_hl)
    loss = F.cross_entropy(out[:author_num], data.y[:author_num])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr, tensor_slice_hl).argmax(dim=-1)
    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[split]

        acc = 0
        acc_cnt = 0
        for i in range(len(data.y[mask])):
            if data.y[mask][i] <= 0: continue
            acc_cnt += 1
            if pred[mask][i] == data.y[mask][i]:
                acc += 1

        if acc_cnt > 0:
            acc /= acc_cnt
        accs.append(float(acc))

    return accs


for epoch in range(1, 10):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
