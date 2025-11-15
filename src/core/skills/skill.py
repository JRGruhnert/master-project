from abc import ABC, abstractmethod
import torch
from src.core.observation import BaseObservation
from src.core.state import BaseState


# class SkillSpace(Enum):
#    Minimal = "Minimal"
#    Normal = "Normal"
#    Full = "Full"
#    Debug = "Debug"


class BaseSkill(ABC):
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
        obs: BaseObservation,
        goal: BaseObservation,
        states: list[BaseState],
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in states:
            if state.name in self.precons.keys():
                value = state.distance_to_skill(
                    obs[state.name],
                    goal[state.name],
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

    @abstractmethod
    def reset(
        self,
    ):
        """Prepare the skill for execution. Before each use."""
        raise NotImplementedError("Subclasses must implement method.")

    @abstractmethod
    def predict(
        self,
        current: BaseObservation,
        goal: BaseObservation,
    ) -> torch.Tensor | None:
        """
        Get the action for the skill.
        """
        raise NotImplementedError("Subclasses must implement method.")

    @abstractmethod
    def _to_skill_format(
        self,
        current: BaseObservation,
        goal: BaseObservation,
    ) -> dict:
        """
        Serialize the skill to a dictionary format.
        """
        raise NotImplementedError("Subclasses must implement method.")


class EmptySkill(BaseSkill):
    def __init__(self):
        super().__init__(name="EmptySkill", id=-1)

    def reset(self):
        pass

    def predict(
        self,
        current: BaseObservation,
        goal: BaseObservation,
    ) -> torch.Tensor | None:
        return None

    def _to_skill_format(
        self,
        current: BaseObservation,
        goal: BaseObservation | None = None,
    ) -> dict:
        return {
            "skill_name": self.name,
            "skill_id": self.id,
            "parameters": {},
        }
