from enum import Enum
from functools import cached_property
import torch
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from torch_geometric.data import Batch, HeteroData
from src.core.networks.layers.encoder import (
    QuaternionEncoder,
    ScalarEncoder,
    PositionEncoder,
)
from src.core.observation import BaseObservation
from src.core.state import BaseState
from src.core.skills.skill import BaseSkill
from tapas_gmm.utils.select_gpu import device
from loguru import logger


class PPOType(Enum):
    ACTOR = "actor"
    CRITIC = "critic"


class ActorCriticBase(nn.Module, ABC):

    def __init__(
        self,
        states: list[BaseState],
        skills: list[BaseSkill],
    ):
        super().__init__()
        self.is_eval_mode = False
        self.states = states
        self.skills = skills
        self.dim_states = len(states)
        self.dim_skills = len(skills)
        self.dim_encoder = 32
        self.encoder_obs = nn.ModuleDict(
            {
                "Euler": PositionEncoder(self.dim_encoder),
                "Quat": QuaternionEncoder(self.dim_encoder),
                "Range": ScalarEncoder(self.dim_encoder),
                "Bool": ScalarEncoder(self.dim_encoder),
                "Flip": ScalarEncoder(self.dim_encoder),
            }
        )

        self.encoder_goal = nn.ModuleDict(
            {
                "Euler": PositionEncoder(self.dim_encoder),
                "Quat": QuaternionEncoder(self.dim_encoder),
                "Range": ScalarEncoder(self.dim_encoder),
                "Bool": ScalarEncoder(self.dim_encoder),
                "Flip": ScalarEncoder(self.dim_encoder),
            }
        )

    def _create_encoders_for_states(self, states: list[BaseState]) -> nn.ModuleDict:
        """Create encoders dynamically based on the actual state types present"""
        unique_types = set(state.type_str for state in states)

        encoder_dict = {}
        for type_str in unique_types:
            encoder_dict[type_str] = self._get_encoder_for_type(type_str)

        return nn.ModuleDict(encoder_dict)

    def _get_encoder_for_type(self, type_str: str) -> nn.Module:
        """Factory method to create appropriate encoder for each state type"""
        encoder_mapping = {
            "Euler": lambda: PositionEncoder(self.dim_encoder),
            "Quat": lambda: QuaternionEncoder(self.dim_encoder),
            "Range": lambda: ScalarEncoder(self.dim_encoder),
            "Bool": lambda: ScalarEncoder(self.dim_encoder),
            "Flip": lambda: ScalarEncoder(self.dim_encoder),
            # Need to find a way to store these mapping externally
        }

        if type_str in encoder_mapping:
            return encoder_mapping[type_str]()
        else:
            raise ValueError(f"Unknown state type: {type_str}")

    def eval(self):
        super().eval()  # Call PyTorch's nn.Module.eval() instead of iterating manually
        self.is_eval_mode = True
        logger.info("Network set to evaluation mode.")

    @abstractmethod
    def forward(
        self,
        obs: list[BaseObservation],
        goal: list[BaseObservation],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def to_batch(
        self,
        obs: list[BaseObservation],
        goal: list[BaseObservation],
    ):
        pass

    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
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
        obs: list[BaseObservation],
        goal: list[BaseObservation],
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy


class BaselineBase(ActorCriticBase):

    def state_type_dict_values(
        self,
        x: BaseObservation,
    ) -> dict[str, torch.Tensor]:
        # TODO: MAKE IT DYNAMIC
        grouped: dict[str, list] = {
            "Euler": [],
            "Quat": [],
            "Range": [],
            "Bool": [],
            "Flip": [],
        }
        for state in self.states:
            value = state.value(x.top_level_observation[state.name])
            grouped[state.type_str].append(value)
        return {
            t: torch.stack(vals).float()
            for t, vals in grouped.items()
            if vals  # only include non-empty
        }

    def to_batch(
        self,
        obs: list[BaseObservation],
        goal: list[BaseObservation],
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
    def to_data(self, obs: BaseObservation, goal: BaseObservation) -> HeteroData:
        pass

    def to_batch(
        self,
        obs: list[BaseObservation],
        goal: list[BaseObservation],
    ) -> Batch:
        data = []
        for o, g in zip(obs, goal):
            data.append(self.to_data(o, g))
        return Batch.from_data_list(data)

    def encode_states(
        self, obs: BaseObservation, goal: BaseObservation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_encoded = [
            self.encoder_obs[state.type_str](
                obs.top_level_observation[state.name].to(device)
            )
            for state in self.states
        ]
        goal_encoded = [
            self.encoder_goal[state.type_str](
                goal.top_level_observation[state.name].to(device)
            )
            for state in self.states
        ]
        obs_tensor = torch.stack(obs_encoded, dim=0)  # [num_states, feature_size]
        goal_tensor = torch.stack(goal_encoded, dim=0)  # [num_states, feature_size]
        return obs_tensor, goal_tensor

    def skill_state_distances(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for skill in self.skills:
            distances = skill.distances(obs, goal, self.states, pad, sparse)
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
        current: BaseObservation,
        goal: BaseObservation,
        pad: bool = True,
        sparse: bool = False,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(
            current, goal, pad, sparse
        )  # [T, S, 1 or 2]
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
        current: BaseObservation,
        goal: BaseObservation,
        pad: bool = True,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, goal, pad)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        return edge_attr
