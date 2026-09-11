from typing import Sequence

import torch
from torch import nn


class FiLM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.generator = nn.Linear(dim, 2 * dim)

    def params(self, cond):
        """The raw ``(gamma, beta)`` for this conditioning input."""
        return self.generator(cond).chunk(2, dim=-1)

    def forward(self, x, cond):
        gamma, beta = self.params(cond)
        return (1.0 + gamma) * x + beta


class FiLMStack(nn.Module):
    def __init__(self, dim: int, names: Sequence[str], sites: Sequence[str]):
        super().__init__()
        self.generators = nn.ModuleDict({name: FiLM(dim) for name in names})
        self.scales = nn.ParameterDict(
            {site: nn.Parameter(torch.ones(2, 1)) for site in sites}
        )

    def forward(
        self, x: torch.Tensor, conds: dict[str, torch.Tensor], site: str
    ) -> torch.Tensor:
        a, b = self.scales[site]
        for name, cond in conds.items():
            gamma, beta = self.generators[name].params(cond)
            x = (1.0 + a * gamma) * x + b * beta
        return x
