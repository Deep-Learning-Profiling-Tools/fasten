from typing import Union, Tuple, Optional
from torch_geometric.typing import OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import RGCNConv

from fasten import TensorSlice, TensorSliceTile
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
    TILE_SIZE = 2

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True,
                 tile_size: int = TILE_SIZE, **kwargs):
        super(FastenRGCNConv, self).__init__(in_channels, out_channels, num_relations,
                                             num_bases, num_blocks, aggr, root_weight, bias, **kwargs)

        self.tile_size = tile_size
        self.weight_type = TensorSlice(self.num_relations)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index, edge_type: TensorSlice):
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
            for h_type in TensorSliceTile(edge_type, self.tile_size):
                edge_slice = slice(h_type.start, h_type.stop)
                # Compute relative offset
                h_type.slices[:, 1:3] -= h_type.start
                out = out + self.propagate(edge_index[:, edge_slice], x=x_l, size=size,
                                           edge_type=h_type, weight=weight)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_type: TensorSlice, weight: Tensor, index: Tensor) -> Tensor:
        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            raise RuntimeError(
                'Block-decomposition is not supported by fasten yet')
        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + index
                return weight.view(-1, self.out_channels)[weight_index]

            print('edge_type')
            print(edge_type.slices)
            return ops.bmm(x_j, edge_type, weight, self.weight_type)

    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        # Compute normalization in separation for each `edge_type`.
        # if self.aggr == 'mean':
        #    norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
        #    norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
        #    norm = torch.gather(norm, 1, edge_type.view(-1, 1))
        #    norm = 1. / norm.clamp_(1.)
        #    inputs = norm * inputs

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
