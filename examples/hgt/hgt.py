import argparse
import os.path as osp
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

import torch_geometric
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.datasets import DBLP, HGBDataset
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils.sparse import index2ptr

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn import FastenHGTConv

from triton.testing import do_bench

torch.backends.cuda.matmul.allow_tf32 = True
torch_geometric.backend.use_segment_matmul = True

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu',
                    choices=['cpu', 'cuda'])
parser.add_argument('--mode', type=str, default='pyg',
                    choices=['pyg', 'fasten'])
parser.add_argument('--example', type=str, default='dblp',
                    choices=['dblp', 'freebase'])
parser.add_argument('--hidden_size', type=int, default=32)
parser.add_argument('--profile', type=str, default='none',
                    choices=['none', 'profile', 'benchmark'])
args = parser.parse_args()

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
device = torch.device(args.device)


def ptr_to_tensor_slice(ptr: List, data: Tensor = None, is_sorted: bool = False) -> Tuple[TensorSlice, List]:

    assert ptr is not None
    slices = [slice(ptr[i], ptr[i + 1]) for i in range(len(ptr) - 1)]
    types = torch.zeros((ptr[-1],), dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = i
    tensor_slice = compact_tensor_types(data=data, types=types, is_sorted=is_sorted, device=device)
    return tensor_slice, slices


def tensor_slice_gen(data, num_heads) -> Tuple[TensorSlice, Tensor, TensorSlice, List]:

    # Generating tensor_slice for HeteroDictLinear
    ptr = [0]
    for key, _ in data.x_dict.items():
        ptr.append(ptr[-1] + data.x_dict[key].shape[0])
    tensor_slice_hdl, slices = ptr_to_tensor_slice(ptr, is_sorted=True)
    slices_hdl = slices

    # Generating tensor_slice for HeteroLinear
    edge_types = data.metadata()[1]
    num_edge_types = len(edge_types)
    H = num_heads   # No of heads
    type_list = []
    edge_map = {edge_type: i for i, edge_type in enumerate(data.metadata()[1])}

    for key, _ in data.edge_index_dict.items():
        N = data.x_dict[key[0]].shape[0]
        edge_type_offset = edge_map[key]
        type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(1, N) * num_edge_types + edge_type_offset
        type_list.append(type_vec)

    type_vec = torch.cat(type_list, dim=1).flatten()
    num_types = H * len(edge_types)
    ptr = index2ptr(type_vec, num_types)
    tensor_slice_hl, _ = ptr_to_tensor_slice(ptr, is_sorted=True)

    return tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        if args.example == "dblp":
            self.node_out = "author"
        else:
            self.node_out = "book"

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict[self.node_out])


class FastenHGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = FastenHGTConv(hidden_channels, hidden_channels, data.metadata(),
                                 num_heads, group='sum', engine=Engine.TRITON)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)
        if args.example == "dblp":
            self.node_out = "author"
        else:
            self.node_out = "book"

    def forward(self, x_dict, edge_index_dict, tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict=x_dict, edge_index_dict=edge_index_dict, tensor_slice_hl=tensor_slice_hl,
                          type_vec=type_vec, tensor_slice_hdl=tensor_slice_hdl, slices_hdl=slices_hdl)

        return self.lin(x_dict[self.node_out])


if args.mode == 'fasten':
    model = FastenHGT(hidden_channels=args.hidden_size, out_channels=out_channels, num_heads=2, num_layers=1)
    data, model = data.to(device), model.to(device)
    tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl = tensor_slice_gen(data, 2)  # last argument num_heads
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict, tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl)

else:
    model = HGT(hidden_channels=args.hidden_size, out_channels=out_channels, num_heads=2, num_layers=1)
    data, model = data.to(device), model.to(device)
    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train(node_out: str):
    with record_function("HGT Train"):
        model.train()
        optimizer.zero_grad()
        with record_function("HGT Inference"):
            if args.mode == 'fasten':
                out = model(data.x_dict, data.edge_index_dict, tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl)
            else:
                out = model(data.x_dict, data.edge_index_dict)

        mask = data[node_out].train_mask
        loss = F.cross_entropy(out[mask], data[node_out].y[mask])
        loss.backward()
        optimizer.step()
        return float(loss)


@torch.no_grad()
def test(node_out: str):
    model.eval()
    if args.mode == 'fasten':
        pred = model(data.x_dict, data.edge_index_dict, tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl).argmax(dim=-1)
    else:
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)

    accs = []
    if args.example == "dblp":
        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = data[node_out][split]
            acc = (pred[mask] == data[node_out].y[mask]).sum() / mask.sum()
            accs.append(float(acc))
    else:
        for split in ['train_mask', 'test_mask']:
            mask = data[node_out][split]
            acc = (pred[mask] == data[node_out].y[mask]).sum() / mask.sum()
            accs.append(float(acc))

    return accs


if args.example == 'dblp':
    node_out = 'author'
else:
    node_out = 'book'

if args.profile == "none":
    for epoch in range(1, 5):
        loss = train(node_out)
        if args.example == "dblp":
            train_acc, val_acc, test_acc = test(node_out)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
        else:
            train_acc, test_acc = test(node_out)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                f'Test: {test_acc:.4f}')

elif args.profile == "profile":
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
        for epoch in range(1, 5):
            train(node_out)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

else: # args.profile == "benchmark"
    def pyg_fn():
        model(data.x_dict, data.edge_index_dict)
    def fasten_fn():
        model(data.x_dict, data.edge_index_dict, tensor_slice_hl, type_vec, tensor_slice_hdl, slices_hdl)
    def train_fn():
        train(node_out)
    fn = pyg_fn if args.mode == "pyg" else fasten_fn
    inference_ms = do_bench(fn)
    train_ms = do_bench(train_fn)
    print(f"{args.mode} inference: {inference_ms} ms")
    print(f"{args.mode} train: {train_ms} ms")
