from torch import nn
import torch


class StateAggregation(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)
        self.pre = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        entity_x: torch.Tensor,
        edge_index: torch.Tensor,
        n_state: int,
    ) -> torch.Tensor:
        src, dst = edge_index
        rows = self.pre(entity_x[src])
        pooled = []
        for slot in range(n_state):
            slot_rows = rows[dst == slot]
            pooled.append(
                self.head(torch.cat([slot_rows.mean(0), slot_rows.amax(0)], dim=-1))
            )
        return torch.stack(pooled)
