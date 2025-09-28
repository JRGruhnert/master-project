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
        reset: bool = False,
    ) -> bool:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @property
    def threshold(self) -> float:
        """Returns the threshold for the state."""
        return 0.05

    @cached_property
    def relative_threshold(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.threshold * (self.upper_bound - self.lower_bound)


class DefaultGoalCondition(GoalCondition):
    """Default goal condition based on Euclidean distance."""

    def distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        reset: bool = False,
    ) -> bool:
        distance = torch.norm(obs - goal).item()
        return distance < self.relative_threshold
