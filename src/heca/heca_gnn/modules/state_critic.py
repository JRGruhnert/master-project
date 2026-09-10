from torch import nn
import torch

from heca.graphs.roles import ROLE_CURRENT, ROLE_GOAL, ROLE_POST, ROLE_PRE


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
    def _mean_max(rows: torch.Tensor, dim: int) -> torch.Tensor:
        if rows.numel() == 0:
            return rows.new_zeros(2 * dim)
        return torch.cat([rows.mean(dim=0), rows.amax(dim=0)], dim=-1)

    @staticmethod
    def _validate(role_ids: torch.Tensor) -> torch.Tensor:
        n_cur = int((role_ids == ROLE_CURRENT).sum())
        n_goal = int((role_ids == ROLE_GOAL).sum())
        if n_cur == 0 and n_goal == 0:
            return role_ids  # no state rows yet
        if n_cur != n_goal:
            raise ValueError
        block = role_ids[-2 * n_cur :]
        if not (
            bool(torch.all(block[0::2] == ROLE_CURRENT))
            and bool(torch.all(block[1::2] == ROLE_GOAL))
        ):
            raise ValueError
        return role_ids

    def forward(
        self,
        entity_x: torch.Tensor,
        role_ids: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dim = entity_x.shape[1]
        role_ids = self._validate(role_ids)
        cur = entity_x[role_ids == ROLE_CURRENT]  # (E, D), entity order
        goal = entity_x[role_ids == ROLE_GOAL]  # (E, D), same entity order
        if cur.numel() == 0:
            return entity_x.new_zeros(1)

        res = cur - goal  # per-entity residual (pairing validated above)
        stats = torch.cat(
            [
                cur.mean(dim=0),
                goal.mean(dim=0),
                res.abs().mean(dim=0),
                self.row_net(res).mean(dim=0),
                self._mean_max(entity_x[role_ids == ROLE_PRE], dim),
                self._mean_max(entity_x[role_ids == ROLE_POST], dim),
            ],
            dim=-1,
        )  # (8D,)
        if memory is None:
            memory = entity_x.new_zeros(1, dim)
        stats = torch.cat([stats, memory.reshape(-1)], dim=-1)  # (9D,)
        self.last_stats = stats.detach()
        return self.head(stats)  # (1,)
