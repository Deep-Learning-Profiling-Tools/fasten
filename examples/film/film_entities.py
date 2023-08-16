import argparse
import os.path as osp

import timemory
import torch
import torch.nn.functional as F
from film_conv import FiLMConv
from sklearn.metrics import f1_score
from timemory.util import marker
from torch.nn import BatchNorm1d
from torch_geometric.datasets import Entities
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

timemory.settings.scientific = False
timemory.settings.flat_profile = False
timemory.settings.timeline_profile = False
timemory.settings.cout_output = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)),
                '..', 'data', args.dataset)
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
num_features = 16
if data.x is None:
    data.x = torch.ones(
        (data.num_nodes, num_features), device=device)
data.edge_type = data.edge_type[edge_mask]
data.train_idx = mapping[:data.train_idx.size(0)]
data.test_idx = mapping[data.train_idx.size(0):]

data = data.to(device)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_relations,
                 dropout=0.0):
        super().__init__()
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            FiLMConv(in_channels, hidden_channels, num_relations=num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(
                FiLMConv(hidden_channels, hidden_channels, num_relations=num_relations))
        self.convs.append(FiLMConv(hidden_channels, out_channels,
                          act=None, num_relations=num_relations))

        self.norms = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(BatchNorm1d(hidden_channels))

    def forward(self, x, edge_index, edge_type):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_type)
        return x


model = Net(in_channels=num_features, hidden_channels=320,
            out_channels=dataset.num_classes, num_layers=4,
            num_relations=dataset.num_relations, dropout=0.1).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    optimizer.zero_grad()
    with marker(["wall_clock"], key="forward"):
        out = model(data.x, data.edge_index, data.edge_type)[data.train_idx]
        loss = criterion(out, data.train_y)
    with marker(["wall_clock"], key="backward"):
        loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()

    pred = model(data.x, data.edge_index, data.edge_type).argmax(dim=-1)
    test_acc = pred[data.test_idx].eq(data.test_y).to(torch.float).mean()

    return test_acc.item()


for epoch in range(1, 100):
    with marker(["wall_clock"], key="train"):
        loss = train()
    with marker(["wall_clock"], key="test"):
        test_acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Test: {test_acc: .4f}')
