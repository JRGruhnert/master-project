from enum import Enum
from functools import cached_property
import torch
import torch.nn as nn
from torch.distributions import Categorical
from abc import ABC, abstractmethod
from torch_geometric.data import Batch, HeteroData
from src.networks.layers.encoder import (
    QuaternionEncoder,
    ScalarEncoder,
    PositionEncoder,
)
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
        self.encoder = nn.ModuleDict(
            {
                "EulerPrecise": PositionEncoder(self.dim_encoder),
                "EulerArea": PositionEncoder(self.dim_encoder),
                "Quat": QuaternionEncoder(self.dim_encoder),
                "Range": ScalarEncoder(self.dim_encoder),
                "Bool": ScalarEncoder(self.dim_encoder),
                "Flip": ScalarEncoder(self.dim_encoder),
            }
        )

    def _create_encoders_for_states(self, states: list[State]) -> nn.ModuleDict:
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

    def encode_by_state_type(
        self,
        x: StateValueDict,
    ) -> StateValueDict:
        """Returns the per state encoded StateValueDict
        \t
        From:
        dict[state_name] = (batch_size, heterogenous state_dim)
        \t
        To:
        dict[state_name] = (batch_size, homogenous encoded_dim)
        """
        encoded = {}
        for state in self.states:
            # Get the encoder for this state type
            encoder = self.encoder[state.type_str]
            # Encode the batched value for this state
            encoded[state.name] = encoder(x[state.name])
        return StateValueDict.from_tensor_dict(encoded, x.batch_size)

    def from_tensor_dict_to_tensor(self, x: StateValueDict) -> torch.Tensor:
        """Returns the per state encoded StateValueDict
        \t
        From:
        dict[state_name] = (batch_size, homogenous dim)
        \t
        To:
        (batch_size, state_num, homogenous encoded_dim)
        """
        tensors = [x[state.name] for state in self.states]
        if tensors[0].ndim == 2:
            # Is batched
            return torch.stack(tensors, dim=1)
        elif tensors[0].ndim == 1:
            # Single sample
            return torch.stack(tensors, dim=0)
        else:
            raise ValueError(
                f"Unexpected tensor dimension: {tensors[0].ndim}, expected 1 or 2."
            )

    def eval(self):
        super().eval()  # Call PyTorch's nn.Module.eval() instead of iterating manually
        self.is_eval_mode = True
        logger.info("Network set to evaluation mode.")

    @abstractmethod
    def forward(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def preprocess(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ):
        pass

    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, goal)
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
        logprob: torch.Tensor = dist.log_prob(action)  # shape: [B]
        return action.detach(), logprob.detach(), value.detach()

    def evaluate(
        self,
        current: StateValueDict,
        goal: StateValueDict,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Categorical]:
        assert current.shape == goal.shape, "Current and Goal have different shapes."
        logits, value = self.forward(current, goal)
        assert logits.shape == (
            current.batch_size[0],
            self.dim_skills,
        ), f"Expected logits shape ({current.batch_size[0]}, {self.dim_skills}), got {logits.shape}"
        assert value.shape == (
            current.batch_size[0],
        ), f"Expected value shape ({current.batch_size[0]},), got {value.shape}"

        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        return action_logprobs, value, dist


class BaselineBase(ActorCriticBase):

    def preprocess(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> torch.Tensor:
        encoded_current = self.encode_by_state_type(current)
        encoded_goal = self.encode_by_state_type(goal)

        tensor_current = self.from_tensor_dict_to_tensor(encoded_current)
        tensor_goal = self.from_tensor_dict_to_tensor(encoded_goal)
        print(tensor_current.shape, tensor_goal.shape)
        if tensor_current.ndim == 3:
            # Batched
            reshaped_current = tensor_current.view(
                tensor_current.shape[0], -1
            )  # [B, S * dim_encoder]
            reshaped_goal = tensor_goal.view(
                tensor_goal.shape[0], -1
            )  # [B, S * dim_encoder]
            return torch.cat([reshaped_current, reshaped_goal], dim=1)
        elif tensor_current.ndim == 2:
            # Single sample
            reshaped_current = tensor_current.view(1, -1)  # [S * dim_encoder]
            reshaped_goal = tensor_goal.view(1, -1)  # [S * dim_encoder]
            return torch.cat([reshaped_current, reshaped_goal], dim=1)
        else:
            raise ValueError(
                f"Unexpected tensor dimension: {tensor_current.ndim}, expected 2 or 3."
            )


class GnnBase(ActorCriticBase, ABC):
    @abstractmethod
    def to_data(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        state_skill: torch.Tensor,
    ) -> HeteroData:
        pass

    def preprocess(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> Batch:
        encoded_current = self.encode_by_state_type(current)
        encoded_goal = self.encode_by_state_type(goal)

        tensor_current = self.from_tensor_dict_to_tensor(encoded_current)
        tensor_goal = self.from_tensor_dict_to_tensor(encoded_goal)
        tensor_state_skill = self.skill_state_distances(current, goal)
        assert tensor_current.shape == tensor_goal.shape
        assert tensor_current.shape[0] == tensor_state_skill.shape[0]

        data = []
        for index in range(tensor_current.shape[0]):
            data.append(
                self.to_data(
                    tensor_current[index],
                    tensor_goal[index],
                    tensor_state_skill[index],
                )
            )
        return Batch.from_data_list(data)

    def skill_state_distances(
        self,
        current: StateValueDict,
        goal: StateValueDict,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        for skill in self.skills:
            distances = skill.distances(current, goal, self.states, pad, sparse)
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
        goal: StateValueDict,
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
        current: StateValueDict,
        goal: StateValueDict,
        pad: bool = True,
    ) -> torch.Tensor:
        dist_matrix = self.skill_state_distances(current, goal, pad)  # [T, S, 2]
        # Now safely get edge attributes for (task, state) pairs: [E, 2]
        edge_attr = dist_matrix[
            self.state_skill_full[1], self.state_skill_full[0]
        ]  # [E, 2]
        return edge_attr
