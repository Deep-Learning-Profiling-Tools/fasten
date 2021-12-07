import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from .rgcn_conv import RGCNConv, FastRGCNConv

from fasten import Ops as fasten_ops
from fasten import TensorSlice
from fasten.nn import FastenRGCNConv

import timemory
from timemory.util import marker

timemory.settings.scientific = False
timemory.settings.flat_profile = False
timemory.settings.timeline_profile = False
timemory.settings.cout_output = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
parser.add_argument('--fasten', action='store_true', default=False)
args = parser.parse_args()

if args.fasten:
    RGCNConv = FastenRGCNConv
else:
    RGCNConv = FastRGCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0]

# BGS and AM graphs are too big to process them in a full-batch fashion.
# Since our model does only make use of a rather small receptive field, we
# filter the graph to only contain the nodes that are at most 2-hop neighbors
# away from any training/test node.
node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx, 2, data.edge_index, relabel_nodes=True)

data.num_nodes = node_idx.size(0)
data.edge_index = edge_index
data.edge_type = data.edge_type[edge_mask]
data.train_idx = mapping[:data.train_idx.size(0)]
data.test_idx = mapping[data.train_idx.size(0):]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = RGCNConv(data.num_nodes, 16, dataset.num_relations,
                              num_bases=30)
        self.conv2 = RGCNConv(16, dataset.num_classes, dataset.num_relations,
                              num_bases=30)

    def forward(self, *args):
        x = F.relu(self.conv1(None, args))
        x = self.conv2(x, args)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    if args.fasten:
        with torch.no_grad():
            edges = fasten_ops.compact(data.edge_index, data.edge_type)
    with marker(["wall_clock", "cuda_event"], key="forward"):
        #print(torch.histc(data.edge_type, bins=46))
        if args.fasten:
            out = model(edges)
        else:
            out = model(data.edge_index, data.edge_type)
        torch.cuda.synchronize()
    with marker(["wall_clock", "cuda_event"], key="loss"):
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        torch.cuda.synchronize()
    with marker(["wall_clock", "cuda_event"], key="backward"):
        loss.backward()
        torch.cuda.synchronize()
    with marker(["wall_clock", "cuda_event"], key="optimize"):
        optimizer.step()
        torch.cuda.synchronize()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
    test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
    return train_acc.item(), test_acc.item()


for epoch in range(1, 10):
    with marker(["wall_clock", "cuda_event"], key="train"):
        loss = train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')

timemory.finalize()
