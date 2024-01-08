import argparse
import os.path as osp
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch import Tensor
from torch.profiler import ProfilerActivity, profile
from torch_geometric.datasets import DBLP, HGBDataset
from torch_geometric.nn import HEATConv, Linear
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr
from triton.testing import do_bench

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn import FastenHEATConv

torch.backends.cuda.matmul.allow_tf32 = True
torch_geometric.backend.use_segment_matmul = True

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'DBLP')
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
parser.add_argument('--mode', type=str, default='pyg',
                    choices=['pyg', 'fasten'])
parser.add_argument('--example', type=str, default='dblp',
                    choices=['dblp', 'freebase'])
parser.add_argument('--profile', type=str, default='none',
                    choices=['none', 'profile', 'benchmark'])
parser.add_argument('--hidden_size', type=int, default=32)
args = parser.parse_args()
device = torch.device(args.device)


# We initialize conference node features with a single one-vector as feature:
if args.example == 'dblp':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
    # We initialize conference node features with a single one-vector as feature:
    dataset = DBLP(path, transform=T.Constant(node_types='conference'))
    out_channels = 4  # 4 class labels
else:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/HGBD')
    transform = T.Compose([T.Constant(value=random.random(),
                                      node_types=['book', 'film', 'music', 'sports', 'people', 'location', 'organization', 'business'])])
    dataset = HGBDataset(path, "Freebase", transform=transform)
    out_channels = 7   # 7 class labels
data = dataset[0]
data = data.to_homogeneous()
# Create ramdom values for edge_attr
data["edge_attr"] = torch.randn((data.edge_index.shape[1], 2))


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
            conv = HEATConv(hidden_channels, hidden_channels, len(torch.unique(data.node_type)), len(torch.unique(data.edge_type)), 5, 2, 6,
                            num_heads, concat=False)
            self.convs.append(conv)

        self.lin_in = Linear(-1, hidden_channels)
        self.lin_out = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, node_type, edge_type, edge_attr):
        x = self.lin_in(x).relu_()

        for conv in self.convs:
            x = conv(x, edge_index, node_type, edge_type, edge_attr)

        return self.lin_out(x)


class FastenHEAT(torch.nn.Module):
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


model = None
if args.mode == 'fasten':
    model = FastenHEAT(hidden_channels=args.hidden_size, out_channels=out_channels, num_heads=2, num_layers=1)
else:
    model = HEAT(hidden_channels=args.hidden_size, out_channels=out_channels, num_heads=2, num_layers=1)

data, model = data.to(device), model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
tensor_slice_hl = tensor_slice_gen(data)


def train():
    model.train()
    optimizer.zero_grad()
    if args.mode == 'fasten':
        out = model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr, tensor_slice_hl)
    else:
        out = model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr)
    node_out = 'author' if args.example == 'dblp' else 'book'
    mask = data[node_out].train_mask
    loss = F.cross_entropy(out[mask], data[node_out].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    if args.mode == 'fasten':
        pred = model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr, tensor_slice_hl).argmax(dim=-1)
    else:
        pred = model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr).argmax(dim=-1)
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


if args.profile == "none":
    for epoch in range(1, 5):
        loss = train()
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

elif args.profile == "profile":
    # warmup
    train()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
        for epoch in range(1, 5):
            train()

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

else:  # args.profile == "benchmark"
    def pyg_fn():
        model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr)

    def fasten_fn():
        model(data.x, data.edge_index, data.node_type, data.edge_type, data.edge_attr, tensor_slice_hl)

    def train_fn():
        train()
    fn = pyg_fn if args.mode == "pyg" else fasten_fn
    inference_ms = do_bench(fn)
    train_ms = do_bench(train_fn)
    print(f"{args.mode} inference: {inference_ms} ms")
    print(f"{args.mode} train: {train_ms} ms")
