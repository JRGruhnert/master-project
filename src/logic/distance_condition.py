from abc import ABC, abstractmethod
import torch

from src.logic.mixin import BoundedMixin, QuaternionMixin


class DistanceCondition(ABC):
    """Abstract base class for skill evaluation strategies."""

    @abstractmethod
    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        goal_tp: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    def expand_if_needed(
        self,
        tensor: torch.Tensor,
        target_shape: torch.Size,
    ) -> torch.Tensor:
        """Expand tensor to target shape if needed."""
        if tensor.shape != target_shape:
            return tensor.expand(target_shape)
        return tensor


class EulerDistanceCondition(DistanceCondition, BoundedMixin):
    """Skill condition based on area matching."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        goal_tp: torch.Tensor,
    ) -> torch.Tensor:
        cx = torch.clamp(current, self.lower_bound, self.upper_bound)
        cy = torch.clamp(goal_tp, self.lower_bound, self.upper_bound)
        nx = self.normalize(cx)
        ny = self.normalize(cy)
        ny = self.expand_if_needed(ny, nx.shape)
        return torch.linalg.norm(nx - ny, dim=-1)


class QuaternionDistanceCondition(DistanceCondition, QuaternionMixin):
    """Skill condition based on quaternion distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        goal_tp: torch.Tensor,
    ) -> torch.Tensor:
        nx = self.normalize_quat(current)
        ny = self.normalize_quat(goal_tp)
        ny = self.expand_if_needed(ny, nx.shape)
        dot = torch.sum(nx * ny, dim=-1)
        dot = torch.clamp(torch.abs(dot), -1.0, 1.0)
        return 2.0 * torch.arccos(dot)


class RangeDistanceCondition(DistanceCondition, BoundedMixin):
    """Skill condition based on range distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        goal_tp: torch.Tensor,
    ) -> torch.Tensor:
        cx = torch.clamp(current, self.lower_bound, self.upper_bound)
        cy = torch.clamp(goal_tp, self.lower_bound, self.upper_bound)
        nx = self.normalize(cx)
        ny = self.normalize(cy)
        ny = self.expand_if_needed(ny, nx.shape)
        return torch.abs(nx - ny)


class BooleanDistanceCondition(DistanceCondition):
    """Skill condition based on boolean distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        goal_tp: torch.Tensor,
    ) -> torch.Tensor:
        ny = self.expand_if_needed(goal_tp, current.shape)
        return torch.abs(current - ny)


class FlipDistanceCondition(DistanceCondition):
    """Skill condition based on flip distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        goal_tp: torch.Tensor,
    ) -> torch.Tensor:
        ny = self.expand_if_needed(goal_tp, current.shape)
        return ny - torch.abs(current - goal)  # Flips distance
