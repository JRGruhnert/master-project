from abc import ABC, abstractmethod
import torch

from hrl.state.logic.mixin import (
    TapasAreaCheckMixin,
    BoundedMixin,
    ThresholdMixin,
)
from hrl.state.logic.target_condition import TargetCondition


# Base class
class EvalCondition(ABC):
    """Abstract base class for success evaluation strategies."""

    @abstractmethod
    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


# Now compose your success conditions using mixins
class PreciseEvalCondition(EvalCondition, ThresholdMixin):
    """Success condition based on precision threshold."""

    def __init__(self, target_condition: TargetCondition):
        ThresholdMixin.__init__(self)
        self.target_condition = target_condition

    def evaluate(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Evaluate success condition based on Euclidean distance."""
        distance = self.target_condition.distance(current, goal, goal)
        success = distance <= self.threshold
        print(
            f"🎯 Eval: obs={current.numpy()[:3]}, goal={goal.numpy()[:3]}, distance={distance:.4f}, success={success}"
        )
        return success


class AreaEvalCondition(EvalCondition, TapasAreaCheckMixin, BoundedMixin):
    """Success condition based on area matching."""

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        return self.check_area_states(obs, goal)


class IgnoreEvalCondition(EvalCondition):
    """Success condition that always returns True (ignores the check)."""

    def evaluate(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        return True
