from abc import ABC, abstractmethod
from functools import cached_property
import torch
import numpy as np


class GoalCondition(ABC):
    """Abstract base class for goal evaluation strategies."""

    @abstractmethod
    def distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


class IgnoreGoalCondition(GoalCondition):
    """Default goal condition based on Euclidean distance."""

    def distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        return True  # Always return True, ignoring the goal condition
