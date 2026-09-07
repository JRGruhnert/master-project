from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn
from torch.distributions import Categorical
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, GINConv

from heca.misc import hardware
from heca.misc.base import Configurable


def _make_gnn_mlp(dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim, dim))
    return nn.Sequential(*layers)


class StepMixBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINEConv(nn=self.nn, edge_dim=8)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.conv(x, edge_index, edge_attr) + x


class TapasBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINConv(nn=self.nn)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x


class SummaryBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.nn = _make_gnn_mlp(dim, num_layers)
        self.conv = GINEConv(nn=self.nn, edge_dim=7)

    def forward(
        self,
        x_entity: torch.Tensor,
        x_option: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv((x_entity, x_option), edge_index, edge_attr)
