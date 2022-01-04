import torch
from heat_conv import HEATConv


def test_small():
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]])
    edge_attr = torch.randn((4, 2))
    node_type = torch.tensor([0, 0, 1, 2])
    edge_type = torch.tensor([0, 2, 1, 2])

    conv = HEATConv(in_channels=8, out_channels=16, num_node_types=3,
                    num_edge_types=3, edge_type_emb_dim=5, edge_dim=2,
                    edge_attr_emb_dim=6, heads=2, concat=True)
    out = conv(x, edge_index, node_type, edge_type, edge_attr)
    assert out.size() == (4, 32)


def test_large():
    x = torch.randn(1000, 32)
    edge_index = torch.randint(low=0, high=1000, size=(2, 4000))
    edge_attr = torch.randn((4000, 4))
    node_type = torch.randint(low=0, high=20, size=(1000,))
    edge_type = torch.randint(low=0, high=20, size=(4000,))

    conv = HEATConv(in_channels=32, out_channels=64, num_node_types=20,
                    num_edge_types=20, edge_type_emb_dim=5, edge_dim=4,
                    edge_attr_emb_dim=6, heads=2, concat=True)
    out = conv(x, edge_index, node_type, edge_type, edge_attr)
    assert out.size() == (1000, 128)


test_small()
test_large()
