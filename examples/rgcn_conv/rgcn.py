import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Entities
from torch_geometric.utils import k_hop_subgraph
from fasten import Ops as ops

import timemory
from timemory.util import marker

timemory.settings.scientific = False
timemory.settings.flat_profile = False
timemory.settings.timeline_profile = False
timemory.settings.cout_output = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
parser.add_argument('--mode', type=str, default='fast',
                    choices=['origin', 'fast', 'fasten'])
parser.add_argument('--profile', action='store_true', default=False)
args = parser.parse_args()

if args.profile is True:
    from rgcn_conv import RGCNConv, FastRGCNConv, FastenRGCNConv
else:
    from fasten.nn import FastenRGCNConv
    from torch_geometric.nn.conv import RGCNConv, FastRGCNConv

if args.mode == 'fast':
    RGCNConv = FastRGCNConv
elif args.mode == 'fasten':
    RGCNConv = FastenRGCNConv
else:
    RGCNConv = RGCNConv

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

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    if args.mode == 'fasten':
        with marker(["wall_clock"], key="compact"):
            with torch.no_grad():
                edge_index, edge_type = ops.compact(
                    data.edge_index, data.edge_type, type_dim=1)
    else:
        edge_index, edge_type = data.edge_index, data.edge_type
    with marker(["wall_clock"], key="forward"):
        # print(torch.histc(data.edge_type, bins=46))
        out = model(edge_index, edge_type)
        torch.cuda.synchronize()
    with marker(["wall_clock"], key="loss"):
        loss = F.nll_loss(out[data.train_idx], data.train_y)
        torch.cuda.synchronize()
    with marker(["wall_clock"], key="backward"):
        loss.backward()
        torch.cuda.synchronize()
    with marker(["wall_clock"], key="optimize"):
        optimizer.step()
        torch.cuda.synchronize()
    return loss.item()


@ torch.no_grad()
def test():
    if args.mode == 'fasten':
        with marker(["wall_clock"], key="compact"):
            with torch.no_grad():
                edge_index, edge_type = ops.compact(
                    data.edge_index, data.edge_type, type_dim=1)
    else:
        edge_index, edge_type = data.edge_index, data.edge_type
    model.eval()
    pred = model(edge_index, edge_type).argmax(dim=-1)
    train_acc = pred[data.train_idx].eq(data.train_y).to(torch.float).mean()
    test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()
    return train_acc.item(), test_acc.item()


for epoch in range(1, 10):
    with marker(["wall_clock"], key="train"):
        loss = train()
    with marker(["wall_clock"], key="test"):
        train_acc, test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')

if args.profile is True:
    print(torch.cuda.memory_summary())

timemory.finalize()
