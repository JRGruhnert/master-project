from abc import ABC
from enum import Enum


class StateSuccess(ABC):
    def evaluate(self, current: float, goal: float) -> bool:
        """Evaluates if the current state is successful compared to the goal."""
        raise NotImplementedError("Must be implemented by subclasses.")


class AreaSuccess(StateSuccess):
    def evaluate(self, current: float, goal: float) -> bool:
        """Checks if the current state is within a certain area of the goal."""
        return abs(current - goal) < 0.1  # Example threshold
