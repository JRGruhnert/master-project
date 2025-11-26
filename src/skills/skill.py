from abc import ABC, abstractmethod
import torch
import numpy as np
from src.observation.observation import StateValueDict
from src.states.state import State


class Skill(ABC):
    def __init__(
        self,
        name: str,
        id: int,
    ):
        self._name: str = name
        self._id: int = id
        self._precons: dict[str, torch.Tensor] = {}
        self._postcons: dict[str, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def id(self) -> int:
        return self._id

    @property
    def precons(self) -> dict[str, torch.Tensor]:
        return self._precons

    @property
    def postcons(self) -> dict[str, torch.Tensor]:
        return self._postcons

    def distances(
        self,
        current: StateValueDict,
        goal: StateValueDict,
        states: list[State],
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in states:
            if state.name in self.precons.keys():
                value = state.distance_to_skill(
                    current[state.name],
                    goal[state.name],
                    self.precons[state.name],
                )
                value = torch.concat([value, torch.tensor([0.0])]) if pad else value
                # 0.0 pad for tasks parameters
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
                # 1.0 pad for non-task parameters
            task_features.append(value)
        return torch.stack(task_features, dim=0)

    @abstractmethod
    def reset(self, *args, **kwargs):
        """Prepare the skill for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray | None:
        """
        Get the next action for the skill.
        """
        raise NotImplementedError("Subclasses must implement method.")
