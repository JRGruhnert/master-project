from abc import ABC, abstractmethod
import torch
from hrl.env.observation import EnvironmentObservation
from hrl.common.state import State


# class SkillSpace(Enum):
#    Minimal = "Minimal"
#    Normal = "Normal"
#    Full = "Full"
#    Debug = "Debug"


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
        self.control_duration = -1
        self.predict_as_batch = False
        self.current_step = -1  # Will be increased at first prediction
        self.predictions = None

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
        obs: EnvironmentObservation,
        goal: EnvironmentObservation,
        states: list[State],
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in states:
            if state.name in self.precons.keys():
                value = state.distance_to_skill(
                    obs.states[state.name],
                    goal.states[state.name],
                    self.precons[state.name],
                )
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
                # 0.0 pad for tasks parameters
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
                # 1.0 pad for non-task parameters
            task_features.append(value)
        return torch.stack(task_features, dim=0)

    def reset(
        self,
        predict_as_batch: bool,
        control_duration: int,
    ):
        """Prepare the skill for execution. Before each use."""
        self.control_duration = control_duration
        self.predict_as_batch = predict_as_batch
        self.current_step = -1  # Will be increased at first prediction
        self.predictions = None

    @abstractmethod
    def predict(
        self,
        current: EnvironmentObservation,
        goal: EnvironmentObservation,
    ) -> torch.Tensor | None:
        """
        Get the action for the skill.
        """
        raise NotImplementedError("Subclasses must implement predict method.")

    @abstractmethod
    def to_skill_format(
        self,
        current: EnvironmentObservation,
        goal: EnvironmentObservation,
        states: list[State],
    ) -> dict:
        """
        Serialize the skill to a dictionary format.
        """
        raise NotImplementedError("Subclasses must implement to_format method.")
