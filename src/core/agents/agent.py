from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.observation import BaseObservation
from src.core.skills.skill import BaseSkill


@dataclass
class AgentConfig:
    batch_size: int = 2048
    eval: bool = False


class BaseAgent(ABC):
    @abstractmethod
    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
    ) -> BaseSkill:
        """Select an action given the current observation and goal observation."""
        raise NotImplementedError("Act method not implemented yet.")

    @abstractmethod
    def feedback(self, reward: float, terminal: bool) -> bool:
        """Pass feedback from the environment. Returns True if the buffer reached the targeted batch size."""
        raise NotImplementedError("Feedback method not implemented yet.")

    @abstractmethod
    def learn(self) -> bool:
        """Perform learning update. Returns True if training should stop. (Plateau reached)"""
        raise NotImplementedError("Learn method not implemented yet.")

    @abstractmethod
    def save(self, tag: str = ""):
        raise NotImplementedError("Save method not implemented yet.")

    @abstractmethod
    def load(self):
        raise NotImplementedError("Load method not implemented yet.")
