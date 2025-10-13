from abc import ABC, abstractmethod
import torch

from src.core.logic.mixin import (
    AreaMixin,
    BoundedMixin,
    ThresholdMixin,
)
from src.core.logic.distance_condition import DistanceCondition


# Base class
class EvalCondition(ABC):
    """Abstract base class for success evaluation strategies."""

    @abstractmethod
    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


# Now compose your success conditions using mixins
class PreciseEvalCondition(EvalCondition, ThresholdMixin):
    """Success condition based on precision threshold."""

    def __init__(self, condition: DistanceCondition):
        ThresholdMixin.__init__(self)
        self.condition = condition

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.condition.distance(current, goal, goal)
        return distance <= self.threshold


class AreaEvalCondition(EvalCondition, AreaMixin):
    """Success condition based on area matching."""

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return self.check_area_similarity(current, goal)


class IgnoreEvalCondition(EvalCondition):
    """Success condition that always returns True (ignores the check)."""

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
