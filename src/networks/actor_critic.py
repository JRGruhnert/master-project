from collections import defaultdict
from enum import Enum
from functools import cached_property
import torch
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from torch_geometric.data import Batch, HeteroData
from src.networks.layers.encoder import StateEncoder
from src.observation.observation import StateValueDict
from src.states.state import State
from src.skills.skill import Skill
from src.hardware import device
from loguru import logger


class PPOType(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class ActorCriticBase(nn.Module, ABC):

    def __init__(
        self,
        states: list[State],
        skills: list[Skill],
    ):
        super().__init__()
        self.is_eval_mode = False
        self.states = states
        self.skills = skills
        self.dim_states = len(states)
        self.dim_skills = len(skills)
        self.dim_encoder = 32

        input_dims = {state.type: state.size for state in states}

        self.encoders = nn.ModuleDict(
            {
                type_str: StateEncoder(input_dim, self.dim_encoder)
                for type_str, input_dim in input_dims.items()
            }
        )

    def eval(self):
        super().eval()  # Call PyTorch's nn.Module.eval() instead of iterating manually
        self.is_eval_mode = True
        logger.info("Network set to evaluation mode.")

    @abstractmethod
    def forward(
        self,
        obs: list[StateValueDict],
        goal: list[StateValueDict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def to_batch(
        self,
        obs: list[StateValueDict],
        goal: list[StateValueDict],
    ):
        pass

    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward([obs], [goal])
        assert logits.shape == (
            1,
            self.dim_skills,
        ), f"Expected logits shape ({1}, {self.dim_skills}), got {logits.shape}"
        assert value.shape == (1,), f"Expected value shape ({1},), got {value.shape}"

        dist = Categorical(logits=logits)
        if self.is_eval_mode:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()  # shape: [B]
        logprob = dist.log_prob(action)  # shape: [B]
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(
        self,
        obs: list[StateValueDict],
        goal: list[StateValueDict],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Categorical]:
        assert len(obs) == len(goal), "Observation and Goal lists have different sizes."
        logits, value = self.forward(obs, goal)
        assert logits.shape == (
            len(obs),
            self.dim_skills,
        ), f"Expected logits shape ({len(obs)}, {self.dim_skills}), got {logits.shape}"
        assert value.shape == (
            len(obs),
        ), f"Expected value shape ({len(obs)},), got {value.shape}"

        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        return action_logprobs, value, dist

    def encode_states(self, x: StateValueDict) -> torch.Tensor:
        encoded_x = [
            self.encoders[state.type](state.make_input(x[state.name]))
            for state in self.states
        ]
        return torch.stack(encoded_x, dim=0)


class BaselineBase(ActorCriticBase):

    def state_type_dict_values(
        self,
        x: StateValueDict,
    ) -> dict[str, torch.Tensor]:
        """Group state values by their type strings."""
        grouped = defaultdict(list)
        for state in self.states:
            value = state.make_input(x[state.name])
            grouped[state.type].append(value)
        return {k: torch.stack(v).float() for k, v in grouped.items()}

    def to_batch(
        self,
        obs: list[StateValueDict],
        goal: list[StateValueDict],
    ):

        obs_encoded = [self.encode_states(o) for o in obs]
        goal_encoded = [self.encode_states(g) for g in goal]

        tensor_obs = torch.stack(obs_encoded, dim=0).to(device)  # [B, S, D]
        tensor_goal = torch.stack(goal_encoded, dim=0).to(device)  # [B, S, D]

        return tensor_obs, tensor_goal


class GnnBase(ActorCriticBase, ABC):
    @abstractmethod
    def to_data(self, obs: StateValueDict, goal: StateValueDict) -> HeteroData:
        pass

    def to_batch(
        self,
        obs: list[StateValueDict],
        goal: list[StateValueDict],
    ) -> Batch:
        data = []
        for o, g in zip(obs, goal):
            data.append(self.to_data(o, g))
        return Batch.from_data_list(data)

    def skill_state_distances(
        self,
        obs: StateValueDict,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for skill in self.skills:
            distances = skill.distances(obs, self.states, pad, sparse)
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
    def state_skill_full(self) -> torch.Tensor:
        src = (
            torch.arange(self.dim_states)
            .unsqueeze(1)
            .repeat(1, self.dim_skills)
            .flatten()
        )
        dst = torch.arange(self.dim_skills).repeat(self.dim_states)
        return torch.stack([src, dst], dim=0)

    @cached_property
    def state_state_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_states)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def skill_skill_sparse(self) -> torch.Tensor:
        indices = torch.arange(self.dim_skills)
        return torch.stack([indices, indices], dim=0)

    @cached_property
    def state_skill_sparse(self) -> torch.Tensor:
        edge_list = []
        for task_idx, skill in enumerate(self.skills):
            for state_idx, state in enumerate(self.states):
                if state.name in skill.precons.keys():
                    edge_list.append((state_idx, task_idx))
        return torch.tensor(edge_list, dtype=torch.long).t()

    @cached_property
    def skill_to_single(self) -> torch.Tensor:
        indices = torch.arange(self.dim_skills)
        return torch.stack([indices, torch.zeros_like(indices)], dim=0)

    @cached_property
    def single_to_skill(self) -> torch.Tensor:
        indices = torch.arange(self.dim_skills)
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
    def state_skill_01_attr(self) -> torch.Tensor:
        sparse = (
            self.state_skill_sparse[0] * self.dim_skills + self.state_skill_sparse[1]
        )
        full = self.state_skill_full[0] * self.dim_skills + self.state_skill_full[1]
        return torch.isin(full, sparse).float().unsqueeze(-1)

    def state_skill_attr_weighted(
        self,
        current: StateValueDict,
        pad: bool = True,
        sparse: bool = False,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, pad, sparse)  # [T, S, 1 or 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        if sparse:
            # Create mask for valid edges (no -1 in any attribute)
            valid_mask = ~(edge_attr == -1).any(dim=1)
            # Filter edges and attributes
            edge_attr = edge_attr[valid_mask]
        return edge_attr

    def state_skill_attr_weighted_sparse(
        self,
        current: StateValueDict,
        pad: bool = True,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, pad)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        return edge_attr
