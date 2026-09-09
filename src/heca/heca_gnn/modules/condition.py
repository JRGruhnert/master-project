import torch
from torch import nn
from torch_geometric.nn import GINEConv

from heca.heca_gnn.modules.common import _make_gnn_mlp


class ConditionBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINEConv(nn=self.nn, edge_dim=8)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_attr) + x
