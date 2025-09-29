from abc import ABC, abstractmethod
import torch
import numpy as np

from abc import ABC, abstractmethod
from functools import cached_property
import torch
import numpy as np

from hrl.state.logic.mixin import AreaCheckMixin


class SkillCondition(ABC):
    """Abstract base class for skill evaluation strategies."""

    @abstractmethod
    def distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")


class AreaSkillCondition(SkillCondition, AreaCheckMixin):
    """Skill condition based on area matching."""

    def distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        return self._check_area_states(obs, goal)
