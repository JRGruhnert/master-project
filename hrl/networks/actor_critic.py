from enum import Enum
from functools import cached_property
import torch
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from torch_geometric.data import Batch, HeteroData
from hrl.networks.layers.encoder import (
    QuaternionEncoder,
    ScalarEncoder,
    PositionEncoder,
)
from hrl.observation.observation import MPObservation
from hrl.state.state import State, StateType
from hrl.skill.skill import Skill
from tapas_gmm.utils.select_gpu import device
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


class PPOType(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class ActorCriticBase(nn.Module, ABC):

    def __init__(
        self,
        states: list[State],
        tasks: list[Skill],
    ):
        super().__init__()
        self.states = states
        self.tasks = tasks
        self.dim_states = len(states)
        self.dim_tasks = len(tasks)
        self.dim_encoder = 32
        self.encoder_obs = nn.ModuleDict(
            {
                StateType.Euler_Angle.name: PositionEncoder(self.dim_encoder),
                StateType.Quaternion.name: QuaternionEncoder(self.dim_encoder),
                StateType.Range.name: ScalarEncoder(self.dim_encoder),
                StateType.Boolean.name: ScalarEncoder(self.dim_encoder),
                StateType.Flip.name: ScalarEncoder(self.dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                StateType.Euler_Angle.name: PositionEncoder(self.dim_encoder),
                StateType.Quaternion.name: QuaternionEncoder(self.dim_encoder),
                StateType.Range.name: ScalarEncoder(self.dim_encoder),
                StateType.Boolean.name: ScalarEncoder(self.dim_encoder),
                StateType.Flip.name: ScalarEncoder(self.dim_encoder),
            }
        )

    @abstractmethod
    def forward(
        self,
        obs: list[MPObservation],
        goal: list[MPObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def to_batch(
        self,
        obs: list[MPObservation],
        goal: list[MPObservation],
    ):
        pass

    def act(
        self,
        obs: MPObservation,
        goal: MPObservation,
        eval_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward([obs], [goal])
        assert logits.shape == (
            1,
            self.dim_tasks,
        ), f"Expected logits shape ({1}, {self.dim_tasks}), got {logits.shape}"
        assert value.shape == (1,), f"Expected value shape ({1},), got {value.shape}"

        dist = Categorical(logits=logits)
        if eval_mode:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(
        self,
        obs: list[MPObservation],
        goal: list[MPObservation],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(obs) == len(goal), "Observation and Goal lists have different sizes."
        logits, value = self.forward(obs, goal)
        assert logits.shape == (
            len(obs),
            self.dim_tasks,
        ), f"Expected logits shape ({len(obs)}, {self.dim_tasks}), got {logits.shape}"
        assert value.shape == (
            len(obs),
        ), f"Expected value shape ({len(obs)},), got {value.shape}"

        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy

    def state_type_dict_values(
        self,
        x: MPObservation,
    ) -> dict[StateType, torch.Tensor]:
        grouped = {t: [] for t in StateType}
        for state in self.states:
            value = state.value(x.states[state.name])
            grouped[state.type].append(value)
        return {
            t: torch.stack(vals).float()
            for t, vals in grouped.items()
            if vals  # only include non-empty
        }


class BaselineBase(ActorCriticBase):
    def to_batch(
        self,
        obs: list[MPObservation],
        goal: list[MPObservation],
    ):
        obs_dicts = [self.state_type_dict_values(o) for o in obs]
        goal_dicts = [self.state_type_dict_values(g) for g in goal]

        tensor_obs = {
            k: torch.stack([d[k] for d in obs_dicts], dim=0).detach().to(device)
            for k in obs_dicts[0].keys()
        }
        tensor_goal = {
            k: torch.stack([d[k] for d in goal_dicts], dim=0).detach().to(device)
            for k in goal_dicts[0].keys()
        }

        return tensor_obs, tensor_goal


class GnnBase(ActorCriticBase, ABC):
    @abstractmethod
    def to_data(self, obs: MPObservation, goal: MPObservation) -> HeteroData:
        pass

    def to_batch(
        self,
        obs: list[MPObservation],
        goal: list[MPObservation],
    ) -> Batch:
        data = []
        for o, g in zip(obs, goal):
            data.append(self.to_data(o, g))
        return Batch.from_data_list(data)

    def encode_states(
        self, obs: MPObservation, goal: MPObservation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_encoded = [
            self.encoder_obs[state.type.name](obs.states[state.name].to(device))
            for state in self.states
        ]
        goal_encoded = [
            self.encoder_goal[state.type.name](goal.states[state.name].to(device))
            for state in self.states
        ]
        obs_tensor = torch.stack(obs_encoded, dim=0)  # [num_states, feature_size]
        goal_tensor = torch.stack(goal_encoded, dim=0)  # [num_states, feature_size]
        return obs_tensor, goal_tensor

    def task_state_distances(
        self,
        obs: MPObservation,
        goal: MPObservation,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for task in self.tasks:
            distances = task.distances(obs, goal, self.states, pad, sparse)
            features.append(distances)
        return torch.stack(features, dim=0).float()

    @cached_property
    def state_state_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_states)
            .unsqueeze(1)
            .repeat(1, self.dim_states)
            .flatten()
        )
        dst = torch.arange(self.dim_states).repeat(self.dim_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_task_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_states)
            .unsqueeze(1)
            .repeat(1, self.dim_tasks)
            .flatten()
        )
        dst = torch.arange(self.dim_tasks).repeat(self.dim_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_state_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def task_task_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_tasks)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def state_task_sparse(self) -> torch.Tensor:
        edge_list = []
        for task_idx, task in enumerate(self.tasks):
            for state_idx, state in enumerate(self.states):
                if state.name in task.task_parameters_keys:
                    edge_list.append((state_idx, task_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    @cached_property
    def task_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_tasks)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_task(self) -> torch.Tensor:
        indices = torch.arange(self.dim_tasks)
        return torch.stack([torch.zeros_like(indices), indices], dim=0)

    @cached_property
    def state_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_state(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
        return torch.stack([torch.zeros_like(indices), indices], dim=0)

    @cached_property
    def state_state_01_attr(self) -> torch.Tensor:
        return (
            (self.state_state_full[0] == self.state_state_full[1])
            .to(torch.float)
            .unsqueeze(-1)
        )

    @cached_property
    def state_task_01_attr(self) -> torch.Tensor:
        sparse = self.state_task_sparse[0] * self.dim_tasks + self.state_task_sparse[1]
        full = self.state_task_full[0] * self.dim_tasks + self.state_task_full[1]
        return torch.isin(full, sparse).float().unsqueeze(-1)

    def state_task_attr_weighted(
        self,
        current: MPObservation,
        goal: MPObservation,
        pad: bool = True,
        sparse: bool = False,
    ) -> torch.Tensor:
        dist_matrix = self.task_state_distances(
            current, goal, pad, sparse
        )  # [T, S, 1 or 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_task_full[1], self.state_task_full[0]
        ]  # [E, 2]
        if sparse:
            # Create mask for valid edges (no -1 in any attribute)
            valid_mask = ~(edge_attr == -1).any(dim=1)
            # Filter edges and attributes
            edge_attr = edge_attr[valid_mask]
        return edge_attr

    def state_task_attr_weighted_sparse(
        self, current: MPObservation, goal: MPObservation, pad: bool = True
    ) -> torch.Tensor:
        dist_matrix = self.task_state_distances(current, goal, pad)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_task_full[1], self.state_task_full[0]
        ]  # [E, 2]
        return edge_attr
