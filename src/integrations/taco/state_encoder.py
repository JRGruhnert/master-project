import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    def __init__(
        self,
        state_keys: list,
        state_sizes: dict,
        hidden_sizes: list = [256, 256],
        latent_dim: int = 32,
    ):
        super().__init__()
        self.state_keys = state_keys
        self.state_sizes = state_sizes

        input_dim = sum([state_sizes[k] for k in state_keys])

        self.mlp = nn.Sequential(
            nn.Linear(-1, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], latent_dim),
        )

    def forward(self, obs) -> torch.Tensor:
        return self.mlp(obs)
