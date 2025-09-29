from abc import ABC, abstractmethod
from hrl.env.observation import EnvironmentObservation
from hrl.skill.skill import Skill


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(
        self, skill: Skill = None
    ) -> tuple[EnvironmentObservation, EnvironmentObservation]:
        """Resets the environment for a new episode. Returns the initial observation and goal observation."""
        raise NotImplementedError("Reset method not implemented yet.")

    @abstractmethod
    def step(self, skill: Skill):
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")

    @abstractmethod
    def evaluate(self) -> tuple[float, bool]:
        """Evaluate the current state against the goal. Returns reward and terminal flag."""
        raise NotImplementedError("Evaluate method not implemented yet.")
