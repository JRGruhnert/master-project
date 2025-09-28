from abc import ABC, abstractmethod
import torch


class EnvironmentObservation(ABC):
    __slots__ = "_states"

    @property
    def states(self) -> dict[str, torch.Tensor]:
        """Returns the scalar states of the observation."""
        return self._states

    @abstractmethod
    def to_skill_format(
        self,
    ) -> torch.Tensor:
        """Returns the observation in the skill format."""
        raise NotImplementedError(
            "Subclasses must implement the to_skill_format method."
        )
