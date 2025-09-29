from abc import ABC, abstractmethod
import torch
import numpy as np

from hrl.state.logic.mixin import (
    AreaCheckMixin,
    BoundedMixin,
    QuaternionMixin,
    RelThresholdMixin,
    ThresholdMixin,
)


# Base class
class SuccessCondition(ABC):
    """Abstract base class for success evaluation strategies."""

    @abstractmethod
    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


# Now compose your success conditions using mixins
class EulerPrecisionSuccessCondition(SuccessCondition, RelThresholdMixin):
    """Success condition based on precision threshold."""

    def __init__(
        self,
        lower_bound: float | np.ndarray = [-1.0, -1.0, -1.0],
        upper_bound: float | np.ndarray = [1.0, 1.0, 1.0],
        *args,
        **kwargs
    ):
        super().__init__(
            lower_bound=lower_bound, upper_bound=upper_bound, *args, **kwargs
        )

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        return torch.norm(obs - goal).item() <= self.relative_threshold


class ScalarPrecisionSuccessCondition(SuccessCondition, RelThresholdMixin):
    """Success condition for discrete states based on precision threshold."""

    def __init__(
        self,
        lower_bound: float | np.ndarray = 0.0,
        upper_bound: float | np.ndarray = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(
            lower_bound=lower_bound, upper_bound=upper_bound, *args, **kwargs
        )

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        return torch.norm(obs - goal).item() <= self.relative_threshold


class AreaSuccessCondition(AreaCheckMixin, BoundedMixin, SuccessCondition):
    """Success condition based on area matching."""

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        return self._check_area_states(obs, goal)


class QuaternionSuccessCondition(RelThresholdMixin, QuaternionMixin, SuccessCondition):
    """Success condition based on quaternion distance."""

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on quaternion distance."""
        return self._quaternion_distance(obs, goal) <= self.relative_threshold


class BooleanSuccessCondition(SuccessCondition):
    """Success condition for boolean states."""

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        return torch.abs(obs - goal).item() == 0.0


class IgnoreSuccessCondition(SuccessCondition):
    """Success condition that always returns True (ignores the check)."""

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
