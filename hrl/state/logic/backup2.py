from abc import ABC, abstractmethod
import torch
from hrl.state.state import State


class StateLogic(ABC):
    """Abstract base class for distance calculation strategies."""

    @abstractmethod
    def value(
        self,
        obs: torch.Tensor,
    ) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @abstractmethod
    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute the distance between the current and goal states."""
        raise NotImplementedError(
            "Subclasses must implement the compute_distance method."
        )

    @abstractmethod
    def goal_distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @abstractmethod
    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor:
        """Returns the mean of the given tensor values."""
        raise NotImplementedError("Subclasses must implement the make_tp method.")


class EuclideanStateLogic(StateLogic):
    """Distance logic based on Euclidean distance."""

    def compute_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        return torch.norm(current - goal).item()

    def evaluate(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        distance = self.compute_distance(obs, goal)
        # Define a threshold for success (this could be parameterized)
        threshold = 0.05
        return distance <= threshold


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
