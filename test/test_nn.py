import pytest
import torch
import triton
from torch_geometric.nn import RGCNConv
from utils import read_slices_from_csv

from fasten import Engine, TensorSlice, compact_tensor_types
from fasten.nn import FastenRGCNConv

AIFB = read_slices_from_csv('AIFB.csv')
AM = read_slices_from_csv('AM.csv')
BGS = read_slices_from_csv('BGS.csv')
DBLP = read_slices_from_csv('DBLP.csv')
MUTAG = read_slices_from_csv('MUTAG.csv')


def slices_to_tensor(tensor_slice: TensorSlice):
    tensor = torch.zeros((tensor_slice.data_size,), dtype=torch.long, device=tensor_slice.data.device)
    for i in range(len(tensor_slice)):
        s = tensor_slice.get_slice_from_index(i, is_tensor=False)
        t = tensor_slice.get_type_from_index(i, is_tensor=False)
        tensor[s] = t
    return tensor


@pytest.mark.parametrize("engine", [Engine.TRITON, Engine.AUTO, Engine.TORCH])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rgcn(engine: Engine, device: str):
    if device == "cpu" and (engine == Engine.TRITON or engine == Engine.AUTO):
        pytest.skip("Triton does not support CPU inference")
    torch.manual_seed(12345)
    x = torch.randn(4, 4).to(device)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 0, 1, 1, 2, 2, 3],
        [1, 1, 1, 2, 1, 1, 1, 0, 1, 3, 1, 3],
    ]).to(device)
    edge_type = torch.tensor([0, 1, 1, 0, 7, 6, 4, 3, 3, 2, 2, 3]).to(device)
    tensor_slice = compact_tensor_types(types=edge_type, data=edge_index, dim=1, device=device)
    sorted_edge_type = slices_to_tensor(tensor_slice)

    fasten_rgcn_conv = FastenRGCNConv(4, 32, 8, is_sorted=True, aggr='add', engine=engine).to(device)
    rgcn_conv_out = fasten_rgcn_conv(x, edge_index=tensor_slice.data, edge_type=sorted_edge_type)
    fasten_rgcn_conv_out = fasten_rgcn_conv(x, edge_index=tensor_slice.data, edge_type=None, edge_tensor_slice=tensor_slice)

    assert fasten_rgcn_conv_out.shape == rgcn_conv_out.shape
    torch.testing.assert_close(fasten_rgcn_conv_out, rgcn_conv_out, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("slices", [AIFB, AM, BGS, DBLP, MUTAG])
@pytest.mark.parametrize("K", [16, 32, 64])
def test_rgcn_perf(slices: list, K: int):
    torch.manual_seed(12345)
    num_nodes = 10000
    num_features = K
    num_types = len(slices)
    num_edges = sum([s.stop - s.start for s in slices])
    x = torch.randn(num_nodes, num_features, device="cuda").requires_grad_(True)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device="cuda")
    types = torch.zeros((num_edges,), device="cuda", dtype=torch.int)
    rand_types = torch.randperm(num_types, device="cuda", dtype=torch.int)
    for i, s in enumerate(slices):
        types[s] = rand_types[i]
    tensor_slice = compact_tensor_types(data=edge_index, types=types, dim=1, device="cuda")
    sorted_edge_type = slices_to_tensor(tensor_slice)
    rgcn_conv = RGCNConv(num_features, num_features, num_types, is_sorted=True, aggr='add').to("cuda")
    fasten_rgcn_conv = FastenRGCNConv(num_features, num_features, num_types, is_sorted=True, aggr='add').to("cuda")

    # warmup and and get grad
    output = rgcn_conv(x, tensor_slice.data, sorted_edge_type)
    grad_rgcn_conv = torch.randn_like(output)

    def rgcn_conv_fn():
        output = rgcn_conv(x, tensor_slice.data, sorted_edge_type)
        output.backward(grad_rgcn_conv)

    ms = triton.testing.do_bench(rgcn_conv_fn)
    print("Torch time:", ms)

    fasten_output = fasten_rgcn_conv(x, tensor_slice.data, None, tensor_slice)
    grad_fasten_rgcn_conv = torch.randn_like(fasten_output)

    def fasten_rgcn_conv_fn():
        output = fasten_rgcn_conv(x, tensor_slice.data, None, tensor_slice)
        output.backward(grad_fasten_rgcn_conv)

    ms = triton.testing.do_bench(fasten_rgcn_conv_fn)
    print("Fasten time:", ms)
