from torch import nn
import torch


class OptionInteraction(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.unsqueeze(0)
        attended, _ = self.attn(xs, xs, xs)
        return self.norm(xs.squeeze(0) + attended.squeeze(0))
