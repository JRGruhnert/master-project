from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.core.observation import BaseObservation
from src.core.skill import BaseSkill


@dataclass
class AgentConfig:
    eval: bool = False


class BaseAgent(ABC):
    @abstractmethod
    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
    ) -> BaseSkill:
        raise NotImplementedError("Act method not implemented yet.")

    @abstractmethod
    def feedback(self, reward: float, terminal: bool):
        raise NotImplementedError("Feedback method not implemented yet.")

    @abstractmethod
    def learn(self) -> bool:
        raise NotImplementedError("Learn method not implemented yet.")

    @abstractmethod
    def save(self, tag: str = ""):
        raise NotImplementedError("Save method not implemented yet.")

    @abstractmethod
    def load(self):
        raise NotImplementedError("Load method not implemented yet.")
