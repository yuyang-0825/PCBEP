from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor, set_diag

from torch_geometric.nn.conv import MessagePassing
from linear import Linear
# from torch.nn import Linear# torch 1.6.0?
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax

# from ..inits import reset
from typing import Any


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))


class PointTransformerConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, pos_nn: Optional[Callable] = None,
                 attn_nn: Optional[Callable] = None,
                 add_self_loops: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.pos_nn = pos_nn
        if self.pos_nn is None:
            self.pos_nn = Linear(3, out_channels)

        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels, bias=False)
        self.lin_src = Linear(in_channels[0], out_channels, bias=False)
        self.lin_dst = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.pos_nn)
        if self.attn_nn is not None:
            reset(self.attn_nn)
        self.lin.reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def normalized_point_pair_features(self, pos_i: Tensor, pos_j: Tensor, normal_i: Tensor,
                            normal_j: Tensor, radius : float) -> Tensor:
        pseudo = pos_j - pos_i
        angle1 = get_angle(normal_i, pseudo)
        angle2 = get_angle(normal_j, pseudo)
        angle3 = get_angle(normal_i, normal_j)
        
        return torch.cat([
            pseudo, angle1.unsqueeze(dim=1), angle2.unsqueeze(dim=1),angle3.unsqueeze(dim=1)
            # torch.sin(angle1),
            # # torch.cos(angle1),
            # torch.sin(angle2),
            # # torch.cos(angle2),
            # torch.sin(angle3),
            # # torch.cos(angle3)
        ], dim=1)


    def forward(
        self,
        x: Union[Tensor, PairTensor],
        pos: Union[Tensor, PairTensor],
        edge_index: Adj,
        normal: Union[Tensor, PairTensor],
        radius: float,
    ) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            alpha = (self.lin_src(x), self.lin_dst(x))
            x: PairTensor = (self.lin(x), x)
        else:
            alpha = (self.lin_src(x[0]), self.lin_dst(x[1]))
            x = (self.lin(x[0]), x[1])

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if isinstance(normal, Tensor):
            normal: PairTensor = (normal, normal)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: PairTensor, pos: PairTensor, alpha: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos, alpha=alpha, normal=normal,
                             size=None, radius=radius)
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor,
                alpha_i: Tensor, alpha_j: Tensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int],
                normal_i: Tensor, normal_j: Tensor, radius_i: float) -> Tensor:

        msg = self.normalized_point_pair_features(pos_i, pos_j, normal_i, normal_j, radius_i)
        delta = self.pos_nn(msg)
        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        return alpha * (x_j + delta)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')