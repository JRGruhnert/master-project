import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv
from src.integrations.calvin.observation import CalvinObservation as MasterObservation
from src.core.networks.actor_critic import GnnBase, PPOType
from src.core.networks.layers.mlp import (
    GinStandardMLP,
    UnactivatedMLP,
)
from tapas_gmm.utils.select_gpu import device


class GinReadoutNetwork(nn.Module):
    def __init__(
        self,
        dim_features: int,
        dim_skill: int,
        dim_state: int,
        ppo_type: PPOType,
    ):
        super().__init__()
        self.ppo_type = ppo_type
        self.dim_skills = dim_skill
        self.dim_state = dim_state
        self.dim_features = dim_features

        self.state_state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_features,
                hidden_dim=self.dim_features,
            ),
            # edge_dim=1,
        )

        self.state_skill_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.dim_features,
                out_dim=self.dim_features,
                hidden_dim=self.dim_features,
            ),
            edge_dim=2,
        )

        self.actor_gin = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

        self.critic_gin = GINConv(
            nn=UnactivatedMLP(self.dim_features, 1),
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict

        x1 = self.state_state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            # edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.state_skill_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        if self.ppo_type is PPOType.ACTOR:

            logits = self.actor_gin(
                x=(x2, x_dict["actor"]),
                edge_index=edge_index_dict[("task", "task-actor", "actor")],
            )
            return logits.view(-1, self.dim_skills)
        else:
            value = self.critic_gin(
                x=(x2, x_dict["critic"]),
                edge_index=edge_index_dict[("task", "task-critic", "critic")],
            )
            return value.squeeze(-1)


class Gnn(GnnBase):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.actor = GinReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_skill=self.dim_skills,
            ppo_type=PPOType.ACTOR,
        )
        self.critic = GinReadoutNetwork(
            dim_features=self.dim_encoder,
            dim_state=self.dim_states,
            dim_skill=self.dim_skills,
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
        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = torch.zeros(self.dim_skills, self.dim_encoder)
        data["actor"].x = torch.zeros(self.dim_skills, 1)
        data["critic"].x = torch.zeros(1, 1)

        data[("goal", "goal-obs", "obs")].edge_index = self.state_state_sparse
        data[("obs", "obs-task", "task")].edge_index = self.state_skill_full
        # data[("goal", "goal-obs", "obs")].edge_attr = self.cnv.state_state_attr
        data[("obs", "obs-task", "task")].edge_attr = self.state_skill_attr_weighted(
            obs, goal
        )
        data[("task", "task-actor", "actor")].edge_index = self.skill_skill_sparse
        data[("task", "task-critic", "critic")].edge_index = self.skill_to_single
        return data.to(device)
