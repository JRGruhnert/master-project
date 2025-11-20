from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.environments.environment import Environment
from src.modules.storage import Storage
from src.skills.skill import Skill
from src.observation.observation import StateValueDict


@dataclass
class ExperimentConfig:
    pass


class Experiment(ABC):
    """Simple Wrapper over the Calvin Environment to perform experiments."""

    def __init__(self, config: ExperimentConfig, env: Environment, storage: Storage):
        # We sort based on Id for the baseline network to be consistent
        self.config = config
        self.env = env
        self.storage = storage

    @abstractmethod
    def step(self, skill: Skill) -> tuple[StateValueDict, float, bool, bool]:
        """Take a step in the environment using the provided skill.
        Returns the new observation, reward, done flag, and terminal flag."""
        raise NotImplementedError("Step method not implemented yet.")

    @abstractmethod
    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        """Sample a new task from the environment.
        Returns the initial observation and goal."""
        raise NotImplementedError("Sample task method not implemented yet.")

    @abstractmethod
    def metadata(self) -> dict:
        """Return experiment metadata as a dictionary."""
        raise NotImplementedError("Metadata method not implemented yet.")

    def close(self):
        self.env.close()
