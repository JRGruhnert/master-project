from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.observation import BaseObservation
from src.core.skills.skill import BaseSkill


@dataclass
class EnvironmentConfig:
    render: bool = False


class BaseEnvironment(ABC):
    @abstractmethod
    def reset(
        self,
        skill: BaseSkill | None = None,
    ) -> tuple[BaseObservation, BaseObservation]:
        """Resets the environment for a new episode. Returns the initial observation and goal observation."""
        raise NotImplementedError("Reset method not implemented yet.")

    @abstractmethod
    def step(self, skill: BaseSkill):
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")

    @abstractmethod
    def evaluate(self) -> tuple[float, bool]:
        """Evaluate the current state against the goal. Returns reward and terminal flag."""
        raise NotImplementedError("Evaluate method not implemented yet.")
