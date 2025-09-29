from abc import ABC, abstractmethod
import numpy as np
import torch

from hrl.state.logic.mixin import TapasAreaCheckMixin, BoundedMixin, QuaternionMixin


class TargetCondition(ABC):
    """Abstract base class for skill evaluation strategies."""

    @abstractmethod
    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


class EulerDistanceCondition(TargetCondition, BoundedMixin):
    """Skill condition based on area matching."""

    def __init__(
        self,
        lower_bound: np.ndarray = np.array([-1.0, -1.0, -1.0]),
        upper_bound: np.ndarray = np.array([1.0, 1.0, 1.0]),
    ):
        BoundedMixin.__init__(self, lower_bound=lower_bound, upper_bound=upper_bound)

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> bool:
        nx = self._normalize(current)
        ny = self._normalize(tp)
        return torch.linalg.norm(nx - ny)


class QuaternionDistanceCondition(TargetCondition, QuaternionMixin):
    """Skill condition based on quaternion distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self._normalize_quat(current)
        ny = self._normalize_quat(tp)
        dot = torch.clamp(torch.abs(torch.dot(nx, ny)), -1.0, 1.0)
        return 2.0 * torch.arccos(dot)


class RangeDistanceCondition(TargetCondition, BoundedMixin):
    """Skill condition based on range distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        cx = torch.clamp(current, self.lower_bound, self.upper_bound)
        cy = torch.clamp(tp, self.lower_bound, self.upper_bound)
        nx = self._normalize(cx)
        ny = self._normalize(cy)
        return torch.abs(nx - ny).item()


class BooleanDistanceCondition(TargetCondition):
    """Skill condition based on boolean distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        return torch.abs(current - tp).item()


class FlipDistanceCondition(TargetCondition):
    """Skill condition based on flip distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        return (tp - torch.abs(current - goal)).item()  # Flips distance
