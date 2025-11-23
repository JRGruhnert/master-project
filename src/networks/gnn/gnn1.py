import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GATv2Conv, GINConv
from src.observation.calvin import CalvinObservation as MasterObservation
from src.networks.actor_critic import GnnBase
from src.hardware import device


class Gnn(GnnBase):
    def __init__(
        self,
        *args,
        dim_gat_out: int = 64,
        attention_heads: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        task_actor_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        task_critic_mlp = nn.Sequential(
            nn.Linear(self.dim_encoder, self.dim_encoder // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_encoder // 2, 1),
        )

        self.gat1 = GATv2Conv(
            (self.dim_encoder, self.dim_encoder),
            self.dim_states,
            heads=attention_heads,
            concat=False,
            edge_dim=1,
            add_self_loops=True,
        )
        self.gat2 = GATv2Conv(
            (self.dim_states, self.dim_states),
            dim_gat_out,
            heads=attention_heads,
            concat=False,
            # edge_dim=1,
            add_self_loops=False,
        )

        self.actor_gin = GINConv(
            nn=task_actor_mlp,
        )

        self.critic_gin = GINConv(
            nn=task_critic_mlp,
        )

    def forward(
        self,
        obs: list[MasterObservation],
        goal: list[MasterObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch: Batch = self.to_batch(obs, goal)
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        edge_attr_dict = batch.edge_attr_dict
        batch_dict = batch.batch_dict

        x1: torch.Tensor = self.gat1(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
            return_attention_weights=None,
        )
        x2 = F.leaky_relu(x1, negative_slope=0.01)  # Apply LeakyReLU after GNN

        x3 = self.gat2(
            x=(x2, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=None,
            return_attention_weights=None,
        )
        x4 = F.leaky_relu(x3, negative_slope=0.01)  # Apply LeakyReLU after GNN

        logits = self.actor(x2).squeeze(-1)
        task_batch_idx = batch_dict["task"]
        v_feat = self.critic_readout(x2, task_batch_idx)
        value = self.critic_head(v_feat).squeeze(-1)
        return logits, value

    def to_data(self, obs: MasterObservation, goal: MasterObservation) -> HeteroData:
        obs_tensor, goal_tensor = self.encode_states(obs, goal)
        task_tensor = self.task_state_distances(obs).to(device)

        data = HeteroData()
        data["goal"].x = goal_tensor
        data["obs"].x = obs_tensor
        data["task"].x = task_tensor

        data[("goal", "goal-obs", "obs")].edge_index = self.state_state_full.to(device)

        data[("obs", "obs-task", "task")].edge_index = self.state_task_full.to(device)

        data[("goal", "goal-obs", "obs")].edge_attr = self.state_state_01_attr.to(
            device
        )
        data[("obs", "obs-task", "task")].edge_attr = self.state_task_01_attr.to(device)
        return data
