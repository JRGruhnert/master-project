from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


@dataclass
class EnvironmentConfig:
    render: bool = False


class Environment(ABC):
    @abstractmethod
    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        """Resets the environment for a new task. Returns the initial observation and goal observation."""
        raise NotImplementedError("Reset method not implemented yet.")

    @abstractmethod
    def step(self, skill: Skill) -> tuple[StateValueDict, float, bool]:
        """Applies the given skill to the environment. Returns the new observation, reward, and done flag."""
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def close(self):
        """Closes the environment and releases any resources."""
        raise NotImplementedError("Close method not implemented yet.")
