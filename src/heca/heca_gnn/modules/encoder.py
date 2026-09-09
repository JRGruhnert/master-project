from torch import nn
import torch

from torch_geometric.nn import GINConv
from heca.heca_gnn.modules.common import _make_gnn_mlp


class FreeEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x


class StaticEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x


class PrismaticEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x


class RevoluteEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x
