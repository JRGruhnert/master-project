from torch import nn


def _make_gnn_mlp(dim: int, num_layers: int) -> nn.Sequential:
    layers = []
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim, dim))
    return nn.Sequential(*layers)
