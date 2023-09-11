import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch_geometric.datasets import Entities
# from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.utils import index_sort, k_hop_subgraph
from torch_geometric.utils.sparse import index2ptr

from fasten import Engine, TensorSlice, compact_tensor_types, ops
from fasten.nn import FastenRGCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()



# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
path = "/home/kganapa/data/Entities"
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

def tensor_slice_gen(data, num_relations):
    edge_type = data.edge_type
    edge_index= data.edge_index
    if (edge_type[1:] < edge_type[:-1]).any():
        edge_type, perm = index_sort(
            edge_type, max_value=num_relations)
        edge_index = edge_index[:, perm]
    edge_type_ptr = index2ptr(edge_type, num_relations)
    edge_ptr= [slice(edge_type_ptr[i], edge_type_ptr[i + 1]) for i in range(len(edge_type_ptr) - 1)]
    M = sum([s.stop - s.start for s in edge_ptr])
    fake_data = torch.randn((M, 1)).to(device)
    types = torch.zeros(M, dtype=torch.long, device=device, requires_grad=False)
    for i, s in enumerate(edge_ptr):
        types[s] = i
    sorted_data, tensor_slice = compact_tensor_types(fake_data, types, device=device)
    return tensor_slice





class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FastenRGCNConv(data.num_nodes, 16, dataset.num_relations)
        self.conv2 = FastenRGCNConv(16, dataset.num_classes, dataset.num_relations)

    def forward(self, edge_index, edge_type, tensor_slice):
        x = F.relu(self.conv1(None, edge_index, edge_type, tensor_slice))
        x = self.conv2(x, edge_index, edge_type, tensor_slice)
        return F.log_softmax(x, dim=1)


if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

model, data = Net().to(device), data.to(device)
tensor_slice = tensor_slice_gen(data, dataset.num_relations)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.edge_index, data.edge_type, tensor_slice)
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.edge_index, data.edge_type, tensor_slice).argmax(dim=-1)
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

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=False, record_shapes=False) as prof:
     with record_function("RGCN Inference"):
        model(data.edge_index, data.edge_type, tensor_slice)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
