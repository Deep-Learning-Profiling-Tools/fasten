from typing import Optional, Union, Tuple
from torch_geometric.typing import OptTensor, Adj

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter as Param
from torch.nn import Parameter
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from timemory.util import marker

from fasten import Ops as ops
from fasten import TensorSlice, TensorSliceTile, Backend


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (Tensor, Tensor) -> Tensor
    pass


@torch.jit._overload
def masked_edge_index(edge_index, edge_mask):
    # type: (SparseTensor, Tensor) -> SparseTensor
    pass


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, Tensor):
        return edge_index[:, edge_mask]
    else:
        return masked_select_nnz(edge_index, edge_mask, layout='coo')


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set to not :obj:`None`, this layer will
            use the basis-decomposition regularization scheme where
            :obj:`num_bases` denotes the number of bases to use.
            (default: :obj:`None`)
        num_blocks (int, optional): If set to not :obj:`None`, this layer will
            use the block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable

        super(RGCNConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError('Can not apply both basis-decomposition and '
                             'block-diagonal-decomposition at the same time.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        if num_bases is not None:
            self.weight = Parameter(
                torch.Tensor(num_bases, in_channels[0], out_channels))
            self.comp = Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert (in_channels[0] % num_blocks == 0
                    and out_channels % num_blocks == 0)
            self.weight = Parameter(
                torch.Tensor(num_relations, num_blocks,
                             in_channels[0] // num_blocks,
                             out_channels // num_blocks))
            self.register_parameter('comp', None)

        else:
            self.weight = Parameter(
                torch.Tensor(num_relations, in_channels[0], out_channels))
            self.register_parameter('comp', None)

        if root_weight:
            self.root = Param(torch.Tensor(in_channels[1], out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.comp)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """

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

        if isinstance(edge_index, SparseTensor):
            edge_type = edge_index.storage.value()
        assert edge_type is not None

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====

            if x_l.dtype == torch.long and self.num_blocks is not None:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)
                h = self.propagate(tmp, x=x_l, size=size)
                h = h.view(-1, weight.size(1), weight.size(2))
                h = torch.einsum('abc,bcd->abd', h, weight[i])
                out += h.contiguous().view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            for i in range(self.num_relations):
                tmp = masked_edge_index(edge_index, edge_type == i)

                if x_l.dtype == torch.long:
                    out += self.propagate(tmp, x=weight[i, x_l], size=size)
                else:
                    h = self.propagate(tmp, x=x_l, size=size)
                    out = out + (h @ weight[i])

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)


class FastRGCNConv(RGCNConv):
    r"""See :class:`RGCNConv`."""

    @marker(['wall_clock'], 'forward')
    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]],
                edge_index: Adj, edge_type: OptTensor = None):
        """"""
        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

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

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=x_l, edge_type=edge_type, size=size)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    @marker(['wall_clock'], 'message')
    def message(self, x_j: Tensor, edge_type: Tensor, edge_index_j: Tensor) -> Tensor:
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            with marker(["wall_clock"], key="decomposition"):
                weight = (self.comp @ weight.view(self.num_bases, -1)).view(
                    self.num_relations, self.in_channels_l, self.out_channels)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if x_j.dtype == torch.long:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            x_j = x_j.view(-1, 1, weight.size(1))
            return torch.bmm(x_j, weight).view(-1, self.out_channels)

        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype == torch.long:
                with marker(["wall_clock"], key="long"):
                    weight_index = edge_type * weight.size(1) + edge_index_j
                    ret = weight.view(-1, self.out_channels)[weight_index]
                    torch.cuda.synchronize()
                    return ret

            with marker(["wall_clock"], key="bmm"):
                ret = torch.bmm(x_j.unsqueeze(-2),
                                weight[edge_type]).squeeze(-2)
                torch.cuda.synchronize()
                return ret

    @marker(['wall_clock'], 'aggregate')
    def aggregate(self, inputs: Tensor, edge_type: Tensor, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            with marker(["wall_clock"], key="mean"):
                norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
                norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
                norm = torch.gather(norm, 1, edge_type.view(-1, 1))
                norm = 1. / norm.clamp_(1.)
                inputs = norm * inputs
                torch.cuda.synchronize()

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)


class FastenRGCNConv(RGCNConv):
    '''
       fasten's RGCNConv implementation

       Args:
           tile_size: number of relation types to be computed for each iteration.
           We trade memory consumption for potential performance gains.
           The larger the tile_size, the more memory consumed and the lower the execution
           time.
           backend: fasten's execution engine
    '''
    TILE_SIZE = 16

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 num_relations: int,
                 num_bases: Optional[int] = None,
                 num_blocks: Optional[int] = None,
                 aggr: str = 'mean',
                 root_weight: bool = True,
                 bias: bool = True,
                 tile_size: int = TILE_SIZE,
                 backend: Backend = Backend.NATIVE,
                 **kwargs):
        super(FastenRGCNConv, self).__init__(in_channels, out_channels, num_relations,
                                             num_bases, num_blocks, aggr, root_weight, bias, **kwargs)

        self.tile_size = tile_size
        self.backend = backend
        self.comp_type = TensorSlice(self.comp, self.num_relations)
        self.weight_type = TensorSlice(self.weight, self.num_relations)
        self.trainable_weights = False

    @marker(['wall_clock'], 'forward')
    def forward(self, x: Union[OptTensor, Tuple[OptTensor, Tensor]], edge_index, edge_type: TensorSlice):
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            self.trainable_weights = True
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)
        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))

        # propagate_type: (x: Tensor)
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)

        # if self.num_bases is not None:  # Basis-decomposition =================
        #    with marker(["wall_clock"], key="decomposition"):
        #        weight = (self.comp @ weight.view(self.num_bases, -1)).view(
        #            self.num_relations, self.in_channels_l, self.out_channels)
        #        torch.cuda.synchronize()

        if self.num_blocks is not None:  # Block-diagonal-decomposition =====
            raise RuntimeError(
                'Block-decomposition is not supported by fasten yet')
        else:  # No regularization/Basis-decomposition ========================
            for h_type in TensorSliceTile(edge_type, self.tile_size):
                with marker(["wall_clock"], key="decomposition"):
                    comp_slice = slice(h_type.get_type(
                        0), h_type.get_type(-1) + 1)
                    num_types = comp_slice.stop - comp_slice.start
                    weight = (self.comp[comp_slice, :] @ self.weight.view(self.num_bases, -1)).view(
                        num_types, self.in_channels_l, self.out_channels)
                    weight_type = self.weight_type.extract(h_type.types())
                    torch.cuda.synchronize()
                edge_slice = slice(h_type.start, h_type.stop)
                if self.trainable_weights is True:
                    # x=None can avoid an extra index_select
                    out = out + self.propagate(edge_index[:, edge_slice], x=None, size=size,
                                               edge_type=h_type.extract(), weight=weight, weight_type=None)
                else:
                    # Compute relative offset
                    out = out + self.propagate(edge_index[:, edge_slice], x=x_l, size=size,
                                               edge_type=h_type.extract(), weight=weight, weight_type=weight_type)

        root = self.root
        if root is not None:
            out += root[x_r] if x_r.dtype == torch.long else x_r @ root

        if self.bias is not None:
            out += self.bias

        return out

    @marker(['wall_clock'], 'message')
    def message(self, x_j: Tensor, edge_type: TensorSlice, weight: Tensor, weight_type: TensorSlice, edge_index_j: Tensor) -> Tensor:
        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            raise RuntimeError(
                'Block-decomposition is not supported by fasten yet')
        else:  # No regularization/Basis-decomposition ========================
            if self.trainable_weights is True:
                with marker(["wall_clock"], key="long"):
                    weight_index = (
                        edge_type.tensor - edge_type.get_type(0)) * weight.size(1) + edge_index_j
                    ret = weight.view(-1,
                                      self.out_channels).index_select(0, weight_index)
                    torch.cuda.synchronize()
                    return ret

            with marker(["wall_clock"], key="bmm"):
                ret = ops.bmm(x_j, edge_type, weight,
                              weight_type, backend=self.backend)
                torch.cuda.synchronize()
                return ret

    @marker(['wall_clock'], 'aggregate')
    def aggregate(self, inputs: Tensor, edge_type: TensorSlice, index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            with marker(["wall_clock"], key="mean"):
                num_types = edge_type.get_type(-1) - edge_type.get_type(0) + 1
                edge_type_offset = edge_type.tensor - edge_type.get_type(0)
                norm = F.one_hot(edge_type_offset, num_types).to(torch.float)
                norm = scatter(norm, index, dim=0, dim_size=dim_size)[index]
                norm = 1. / torch.gather(norm, 1, edge_type_offset.view(-1, 1))
                inputs = norm * inputs
                torch.cuda.synchronize()

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(self.__class__.__name__,
                                                     self.in_channels,
                                                     self.out_channels,
                                                     self.num_relations)
