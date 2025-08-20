from abc import ABC, abstractmethod
from enum import Enum
from functools import cached_property
from loguru import logger
from master_project.registry.state_type_registry import StateType
import numpy as np
import torch
import json
from pathlib import Path
from typing import List


class StateSuccess(Enum):
    Area = "Area"  # Has to be in the same area
    Precise = "Precise"  # Has to be precise (with threshold)
    Ignore = "Ignore"  # Is not used in Success calculation


class StateType(ABC):
    @classmethod
    def _create_state_by_type(
        cls,
        state_type: StateType,
        name: str,
        success: StateSuccess,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ) -> "State":
        """Factory method using registry"""
        if state_type not in cls._state_registry:
            raise ValueError(f"Unknown state type: {state_type}")

        state_class = cls._state_registry[state_type]
        return state_class(name, state_type, success, lower_bound, upper_bound)

    @classmethod
    def from_json(cls, name: str, json_data: dict) -> "StateType":
        """Create a State instance from JSON data"""
        if (
            "type" not in json_data
            or "space" not in json_data
            or "success" not in json_data
            or "lower_bound" not in json_data
            or "upper_bound" not in json_data
        ):
            raise ValueError(f"Invalid JSON data for State {name}")
        if not isinstance(json_data["lower_bound"], list):
            raise ValueError(f"Invalid JSON data for State {name}")
        if not isinstance(json_data["upper_bound"], list):
            raise ValueError(f"Invalid JSON data for State {name}")

        state_type = StateType(json_data["type"])
        # sub_type = StateSubType(json_data["sub_type"])

        # Prepare common arguments for all state types
        common_args = {
            "name": name,
            "state_type": state_type,
            "success": StateSuccess(json_data["success"]),
            "lower_bound": torch.tensor(json_data["lower_bound"], dtype=torch.float32),
            "upper_bound": torch.tensor(json_data["upper_bound"], dtype=torch.float32),
        }
        # Default implementation for base class or when called directly on subclasses
        return cls._create_state_by_type(**common_args)

    @classmethod
    def from_json_list(cls, state_space: StateSpace) -> List["State"]:
        """Convert a StateSpace to a list of State objects by reading from states.json"""
        # Load the states.json file
        states_json_path = Path(__file__).parent / "data" / "states.json"

        if not states_json_path.exists():
            raise FileNotFoundError(f"States JSON file not found at {states_json_path}")

        with open(states_json_path, "r") as f:
            data: dict = json.load(f)

        # Filter states based on the requested state space
        filtered = []

        for state_key, state_value in data.items():
            # Check if this state belongs to the requested space
            state_space_list = state_value.get("space")
            if state_space_list is None:
                raise ValueError(f"State {state_key} does not have a 'space' defined.")

            if state_space.value in state_space_list:
                state = cls.from_json(state_key, state_value)
                filtered.append(state)

        return filtered

    def __init__(
        self,
        name: str,
        type: StateType,
        success: StateSuccess,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
    ):
        self._name = name
        self._type = type
        self._success = success
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        assert self._lower_bound.shape == self._upper_bound.shape

        if self._type == StateType.Euler_Angle or self._type == StateType.Axis_Angle:
            assert self._lower_bound.shape == self._upper_bound.shape == (3,)
        elif self._type == StateType.Range:
            assert self._lower_bound.shape == self._upper_bound.shape == (1,)
        if self._type != StateType.Euler_Angle and self._type != StateType.Axis_Angle:
            assert (
                success is not StateSuccess.Area
            ), f"State {self._name} cannot have Area based success evaluation, because it is not a position-based state."

    @property
    def name(self) -> str:
        """Returns the StateIdent of the state."""
        return self._name

    @property
    def type(self) -> StateType:
        """Returns the StateType of the state."""
        return self._type

    @property
    def success_mode(self) -> StateSuccess:
        """Returns the StateSuccess of the state."""
        return self._success

    @property
    def lower_bound(self) -> torch.Tensor:
        """Returns the lower bound of the state."""
        return self._lower_bound

    @property
    def upper_bound(self) -> torch.Tensor:
        """Returns the upper bound of the state."""
        return self._upper_bound

    @property
    def threshold(self) -> float:
        """Returns the threshold for the state."""
        return 0.05

    @cached_property
    def relative_threshold(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.threshold * (self.upper_bound - self.lower_bound)

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value of the state as a tensor."""
        raise NotImplementedError("Must be implemented by subclasses.")

    @abstractmethod
    def distance_to_tp(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        raise NotImplementedError("Must be implemented by subclasses.")

    @abstractmethod
    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        raise NotImplementedError("Must be implemented by subclasses.")

    @abstractmethod
    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        # TODO: param: tapas_selection should be removed to generalize this method
        raise NotImplementedError("Must be implemented by subclasses.")

    def evaluate_success_condition(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        surfaces: dict[str, np.ndarray],
        reset: bool = False,
    ) -> bool:
        if self.success_mode == StateSuccess.Area:
            return self.check_area_states(obs, goal, surfaces)
        elif self.success_mode == StateSuccess.Precise:
            return self.distance_to_goal(obs, goal) <= self.threshold
        elif self.success_mode == StateSuccess.Ignore:
            return True
        else:
            raise NotImplementedError(
                f"State Success: {self.success_mode} is not implemented."
            )

    def area_tapas_override(
        self, x: torch.Tensor, surfaces: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Override the area check for TAPAS.
        """
        area = self.check_area(x, surfaces)
        if area == "closed_drawer":
            x[1] -= 0.17  # Drawer Offset
        return x  # Return original point if no area match

    def check_area(
        self, x: torch.Tensor, surfaces: dict[str, torch.Tensor]
    ) -> str | None:
        """
        Check if the point x is in any of the defined areas.
        Returns the name of the area or None if not found.
        """
        for name, (min_corner, max_corner) in surfaces.items():
            box_min = torch.tensor(min_corner)
            box_max = torch.tensor(max_corner)
            if torch.all(x >= box_min) and torch.all(x <= box_max):
                return name
        return None

    def check_area_states(
        self, x: torch.Tensor, y: torch.Tensor, surfaces: dict[str, torch.Tensor]
    ) -> bool:
        area_x = self.check_area(x, surfaces)
        area_y = self.check_area(y, surfaces)
        return area_x == area_y


class LinearState(State):
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a value x to the range [0, 1] based on min and max.
        """
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, self.lower_bound, self.upper_bound)
        return self.normalize(cx)


class DiscreteState(State):
    def value(self, x: torch.Tensor) -> torch.Tensor:
        return x


@State.register_type(StateType.Euler_Angle)
class EulerState(LinearState):

    def distance_to_tp(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self.normalize(current)
        ny = self.normalize(tp)
        return torch.linalg.norm(nx - ny)

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.distance_to_tp(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if not tapas_selection:
            return None  # Tapas didn't select this state
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)


@State.register_type(StateType.Quaternion)
class QuaternionState(State):
    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.linalg.norm(x)
        if x[3] < 0:
            return -x
        return x

    def value(self, x):
        return self.normalize_quat(x)

    def distance_to_tp(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self.normalize_quat(current)
        ny = self.normalize_quat(tp)
        dot = torch.clamp(torch.abs(torch.dot(nx, ny)), -1.0, 1.0)
        return 2.0 * torch.arccos(dot)

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.distance_to_tp(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if not tapas_selection:
            return None  # Tapas didn't select this state
        if reversed:
            return self.quaternion_mean(end)
        return self.quaternion_mean(start)

    def quaternion_mean(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean quaternion using the eigenvector method.
        quaternions: tensor of shape [N, 4] (x, y, z, w)
        Returns: mean quaternion [4] in (x, y, z, w) format
        """
        # Swap to (w, x, y, z) for computation
        quats = quaternions[:, [3, 0, 1, 2]]
        quats = quats / quats.norm(dim=1, keepdim=True)
        A = quats.t() @ quats
        _, eigenvectors = torch.linalg.eigh(A)
        mean_quat = eigenvectors[:, -1]
        # Ensure positive scalar part
        if mean_quat[0] < 0:
            mean_quat = -mean_quat
        # Swap back to (x, y, z, w)
        mean_quat_xyzw = mean_quat[[1, 2, 3, 0]]
        return self.normalize_quat(mean_quat_xyzw)


@State.register_type(StateType.Range)
class RangeState(LinearState):

    def distance_to_tp(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        cx = torch.clamp(current, self.lower_bound, self.upper_bound)
        cy = torch.clamp(tp, self.lower_bound, self.upper_bound)
        nx = self.normalize(cx)
        ny = self.normalize(cy)
        return torch.abs(nx - ny).item()

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.distance_to_tp(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not constant enough


@State.register_type(StateType.Boolean)
class BooleanState(DiscreteState):

    def distance_to_tp(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        return torch.abs(current - tp).item()

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.distance_to_tp(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not constant enough


@State.register_type(StateType.Flip)
class FlipState(DiscreteState):

    def distance_to_tp(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return (tp - torch.abs(current - goal)).item()  # Flips distance

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return torch.abs(current - goal).item()

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state
        return None
