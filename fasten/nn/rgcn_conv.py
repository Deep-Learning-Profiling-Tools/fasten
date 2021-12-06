from typing import Union, Tuple
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
from torch_geometric.nn import RGCNConv

from fasten import TensorSlice
from fasten import Ops as ops


class FastenRGCNConv(RGCNConv):
    '''
       fasten's RGCNConv implementation

       Args:
           tile_size: number of relation types to be computed for each iteration.
           We trade memory consumption for potential performance gains.
           The larger the tile_size, the more memory consumed and the lower the execution
           time.
    '''
    TILE_SIZE = 8

    def __init__(self, **kwargs):
        super(FastenRGCNConv, self).__init__(**kwargs)

        self.tile_size = FastenRGCNConv.TILE_SIZE
        if 'tile_size' in kwargs:
            self.tile_size = kwargs['tile_size']

        if self.num_bases is not None:
            self.weight_types = TensorSlice(self.num_bases)
        elif self.num_blocks is not None:
            self.weight_types = TensorSlice(self.num_relations)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index, edge_types: TensorSlice):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight

        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====
            raise RuntimeError(
                'Block-decomposition is not supported by fasten yet')
        else:  # No regularization/Basis-decomposition ========================
            for i in range(0, self.num_relations, self.tile_size):
                start_relation = i
                end_relation = self.num_relations - 1 if start_relation + \
                    self.tile_size >= self.num_relations else start_relation + self.tile_size - 1
                start_index = edge_types[start_relation, 0].item()
                end_index = edge_types[end_relation - 1, 1].item()
                weight_indices = slice(start_relation, end_relation)
                edge_indices = slice(start_index, end_index)
                tmp = edge_index[:, edge_indices]

                if x_l.dtype == torch.long:
                    out += self.propagate(tmp,
                                          x=weight[weight_indices, x_l], size=size)
                else:
                    h_slice = self.propagate(tmp, x=x_l, size=size)
                    h_types = edge_types.extract(weight_indices)
                    weight_slice = weight[weight_indices, :]
                    weight_types = self.weight_types.extract(weight_indices)
                    out = out + ops.bmm(h_slice, h_types,
                                        weight_slice, weight_types)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return torch.sum(x_j, 0)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
