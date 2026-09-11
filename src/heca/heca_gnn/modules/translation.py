import torch
from torch import nn
from torch_geometric.nn import GINConv
from heca.heca_gnn.modules.common import _make_gnn_mlp


class TranslationBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2):
        super().__init__()
        self.conv = GINConv(nn=_make_gnn_mlp(dim, num_layers))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index) + x
