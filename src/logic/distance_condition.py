from abc import ABC, abstractmethod
import math
import torch


class DistanceCondition(ABC):
    """Abstract base class for skill evaluation strategies."""

    @abstractmethod
    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


class EulerDistanceCondition(DistanceCondition):
    """Skill condition based on area matching."""

    def __init__(self):
        self.max_dist = math.sqrt(3)

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return (torch.linalg.norm(current - x) / self.max_dist).item()


class QuaternionDistanceCondition(DistanceCondition):
    """Skill condition based on quaternion distance."""

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        dot = torch.clamp(torch.abs(torch.dot(current, x)), -1.0, 1.0)
        return (2.0 * torch.arccos(dot) / math.pi).item()


class RangeDistanceCondition(DistanceCondition):
    """Skill condition based on range distance."""

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return torch.abs(current - x).item()


class FlipDistanceCondition(DistanceCondition):
    """Skill condition based on flip distance."""

    def distance(
        self,
        current: torch.Tensor,
        x: torch.Tensor,
    ) -> float:
        return 0.0  # Always return zero distance for flip conditions
