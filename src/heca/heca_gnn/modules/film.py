from torch import nn


class FiLM(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def forward(self, x, cond):
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)
        return gamma * x + beta
