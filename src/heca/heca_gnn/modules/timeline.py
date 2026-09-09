from torch import nn
import torch


class TimelineMemory(nn.Module):
    def __init__(self, dim: int, extra_in: int = 0):
        super().__init__()
        self.gru = nn.GRUCell(dim + extra_in, dim)

    def forward(self, u: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = u.new_zeros(u.shape[0], self.gru.hidden_size)
        return self.gru(u, h)
