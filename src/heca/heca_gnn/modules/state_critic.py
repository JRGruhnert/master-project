from torch import nn
import torch

from heca.graphs.roles import ROLE_POST, ROLE_PRE


class StateCritic(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * hidden_ratio), 16)

        self.row_net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        in_dim = 9 * dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _mean_max(rows: torch.Tensor) -> torch.Tensor:
        return torch.cat([rows.mean(dim=0), rows.amax(dim=0)], dim=-1)

    def forward(
        self,
        entity_x: torch.Tensor,
        role_ids: torch.Tensor,
        cur_idx: torch.Tensor,
        goal_idx: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        cur = entity_x[cur_idx]  # (E, D) — same entity order as goal
        goal = entity_x[goal_idx]  # (E, D)

        res = cur - goal  # per-entity residual (progress toward the goal)
        stats = torch.cat(
            [
                cur.mean(dim=0),
                goal.mean(dim=0),
                res.abs().mean(dim=0),
                self.row_net(res).mean(dim=0),
                self._mean_max(entity_x[role_ids == ROLE_PRE]),
                self._mean_max(entity_x[role_ids == ROLE_POST]),
            ],
            dim=-1,
        )  # (8D,)
        stats = torch.cat([stats, memory.reshape(-1)], dim=-1)  # (9D,)
        self.last_stats = stats.detach()
        return self.head(stats)  # (1,)
