from abc import ABC, abstractmethod
import torch
import numpy as np

from abc import ABC, abstractmethod
from functools import cached_property
import torch
import numpy as np

from hrl.state.logic.mixin import AreaCheckMixin


class SkillCondition(ABC):
    """Abstract base class for skill evaluation strategies."""

    @abstractmethod
    def distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


class AreaSkillCondition(SkillCondition, AreaCheckMixin):
    """Skill condition based on area matching."""

    def distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        return self._check_area_states(obs, goal)


class Euler2State(State):
    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self.normalize(current)
        ny = self.normalize(tp)
        return torch.linalg.norm(nx - ny)

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)


class Quaternion2State(State):
    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.linalg.norm(x)
        if x[3] < 0:
            return -x
        return x

    def value(self, x):
        return self.normalize_quat(x)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self.normalize_quat(current)
        ny = self.normalize_quat(tp)
        dot = torch.clamp(torch.abs(torch.dot(nx, ny)), -1.0, 1.0)
        return 2.0 * torch.arccos(dot)

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)


class Range2State(State):

    def tp_distance(
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

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)


class Boolean2State(State):

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        return torch.abs(current - tp).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)


class Flip2State(State):

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return (tp - torch.abs(current - goal)).item()  # Flips distance

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return torch.abs(current - goal).item()


class EulerAngleLogic(LinearStateLogic):
    """Logic for Euler angle states (3D rotations) - needs bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        threshold: float = 0.05,
    ):
        # Validate that bounds are 3D before calling super
        assert lower_bound.shape == upper_bound.shape == (3,), "Euler angles must be 3D"
        super().__init__(lower_bound, upper_bound, threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute normalized Euclidean distance for Euler angles."""
        nx = self.normalize(current)
        ny = self.normalize(tp)
        return torch.linalg.norm(nx - ny).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)


class QuaternionValueConverter(ValueConverter):
    """Logic for quaternion states (4D rotations) - no bounds needed."""

    def __init__(self, threshold: float = 0.05):
        # Quaternions don't need bounds - they're always unit quaternions
        super().__init__(threshold)

    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion and ensure positive w component."""
        x = x / torch.linalg.norm(x)
        if x[3] < 0:
            return -x
        return x

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.normalize_quat(x)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute quaternion distance using angular metric."""
        nx = self.normalize_quat(current)
        ny = self.normalize_quat(tp)
        dot = torch.clamp(torch.abs(torch.dot(nx, ny)), 0.0, 1.0)
        return 2.0 * torch.arccos(dot).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)


class RangeLogic(LinearStateLogic):
    """Logic for scalar range states - needs bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        threshold: float = 0.05,
    ):
        # Validate that bounds are scalar before calling super
        assert (
            lower_bound.shape == upper_bound.shape == (1,)
        ), "Range states must be scalar"
        super().__init__(lower_bound, upper_bound, threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute normalized distance for range states."""
        cx = torch.clamp(current, self.lower_bound, self.upper_bound)
        cy = torch.clamp(tp, self.lower_bound, self.upper_bound)
        nx = self.normalize(cx)
        ny = self.normalize(cy)
        return torch.abs(nx - ny).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)


class BooleanLogic(DiscreteStateLogic):
    """Logic for boolean states - no bounds needed, just 0/1 values."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)
        # For boolean logic, we can define a relative threshold for consistency checks
        self.relative_threshold = torch.tensor([threshold])

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute boolean distance (0 or 1)."""
        return torch.abs(current - tp).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point if boolean state is consistent."""
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not consistent enough


class FlipLogic(DiscreteStateLogic):
    """Logic for flip states (inverted distance) - no bounds needed."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute flipped distance (inverted)."""
        return (tp - torch.abs(current - goal)).item()  # Flipped distance

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute normal distance to goal."""
        return torch.abs(current - goal).item()

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point if flip occurred."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state detected
        return None
