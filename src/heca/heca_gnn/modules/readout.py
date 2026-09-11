from torch import nn
import torch


class OptionReadout(nn.Module):
    SITE = "actor"

    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(int(dim * hidden_ratio), 16)

        self.norm = nn.LayerNorm(dim)
        self.shared = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, films, conds: dict) -> torch.Tensor:
        shared = films(self.norm(x), conds, self.SITE)
        actor_out = self.actor_head(self.shared(shared))
        return actor_out.view(1, -1)
