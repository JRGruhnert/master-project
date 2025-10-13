import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.nn import GINConv
from src.integrations.calvin.observation import CalvinObservation as MasterObservation
from src.core.networks.actor_critic import GnnBase, PPOType
from src.core.networks.layers.mlp import (
    GinUnactivatedMLP,
    GinStandardMLP,
)
from tapas_gmm.utils.select_gpu import device


class SimpleMeanMaxPoolNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_task: int,
        dim_state: int,
        ppo_type: PPOType,
    ):
        super().__init__()
        self.ppo_type = ppo_type
        self.dim_tasks = dim_task
        self.dim_state = dim_state
        self.dim_features = dim_features

        self.state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_state,
            ),
        )

        self.action_gin = GINConv(
            nn=GinUnactivatedMLP(self.dim_state),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        batch_dict = batch.batch_dict
        task_batch_idx = batch_dict["task"]  # same length as total_task_nodes

        x1 = self.state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.action_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
        )

        if self.ppo_type is PPOType.ACTOR:
            return x2.view(-1, self.dim_tasks)  # [B, dim_tasks]
        else:
            max_pool = global_max_pool(x2, task_batch_idx)  # [B, D]
            mean_pool = global_mean_pool(x2, task_batch_idx)  # [B, D]
            # reduce to a (B, ‑) vector for your critic head:
            pooled = torch.cat(
                [
                    max_pool.max(dim=1).values.unsqueeze(-1),
                    mean_pool.mean(dim=1).unsqueeze(-1),
                ],
                dim=1,
            )  # [B,2]

            return self.critic_head(pooled).squeeze(-1)  # [B]


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.actor = SimpleMeanMaxPoolNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_task=self.dim_tasks,
            ppo_type=PPOType.ACTOR,
        )
        self.critic = SimpleMeanMaxPoolNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_task=self.dim_tasks,
            ppo_type=PPOType.CRITIC,
        )

    def forward(
        self,
        obs: list[MasterObservation],
        goal: list[MasterObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        logits = self.actor(batch)
        value = self.critic(batch)
        return logits, value

    def to_data(self, obs: MasterObservation, goal: MasterObservation) -> HeteroData:
        obs_tensor, goal_tensor = self.encode_states(obs, goal)
        task_tensor = self.task_state_distances(obs)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.state_state_sparse
        data[("obs", "obs-task", "task")].edge_index = self.state_task_sparse
        return data.to(device)
