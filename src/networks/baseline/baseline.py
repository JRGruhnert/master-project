import torch
import torch.nn as nn
from src.networks.actor_critic import BaselineBase
from src.observation.observation import StateValueDict


class Baseline(BaselineBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.combined_feature_dim = self.dim_encoder * self.dim_states * 2

        h_dim1 = self.combined_feature_dim // 2
        h_dim2 = h_dim1 // 2
        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, self.dim_skills),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(self.combined_feature_dim, h_dim1),
            nn.ReLU(),
            nn.Linear(h_dim1, h_dim2),
            nn.ReLU(),
            nn.Linear(h_dim2, 1),
        )

    def forward(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        #
        x = self.preprocess(current, goal)
        # Forward through actor and critic
        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)  # shape: [B]
        return logits, value
