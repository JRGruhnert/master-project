from abc import ABC, abstractmethod
from src.core.skills.skill import BaseSkill
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation


class Experiment(ABC):
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, env: CalvinEnvironment):
        # We sort based on Id for the baseline network to be consistent
        self.env = env

    @abstractmethod
    def step(self, skill: BaseSkill) -> CalvinObservation:
        pass

    @abstractmethod
    def sample(self) -> tuple[CalvinObservation, CalvinObservation]:
        pass

    @abstractmethod
    def evaluate(self) -> tuple[float, bool]:
        pass

    @abstractmethod
    def close(self):
        pass
