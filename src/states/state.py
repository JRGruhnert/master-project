import torch

from src.logic.addon import BaseAddon
from src.logic.distance_condition import (
    DistanceCondition,
)
from src.logic.value_condition import (
    ValueCondition,
)
from src.logic.eval_condition import (
    EvalCondition,
)


class State:

    def __init__(
        self,
        name: str,
        id: int,
        type_str: str,
        size: int,
        normalizer: ValueCondition,
        skill_condition: DistanceCondition,
        goal_condition: DistanceCondition,
        eval_condition: EvalCondition,
        addons: dict[str, BaseAddon],
        eval_normalizer: ValueCondition | None = None,
    ):
        self.name = name
        self.id = id
        self.type = type_str
        self.size = size
        self._normalizer = normalizer
        self._eval_normalizer = (
            eval_normalizer if eval_normalizer is not None else normalizer
        )
        self._skill_condition = skill_condition
        self._goal_condition = goal_condition
        self._eval_condition = eval_condition
        self._addons = addons

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value of the state as a tensor."""
        assert isinstance(x, torch.Tensor), "Input must be torch.Tensor"
        return self._normalizer.make_input(x)

    def distance_to_skill(
        self,
        current: torch.Tensor,
        precon: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        assert isinstance(current, torch.Tensor) and isinstance(
            precon, torch.Tensor
        ), "Inputs must be torch.Tensor"
        current_norm = self._normalizer.value(current)
        precon_norm = self._normalizer.value(precon)
        value = self._skill_condition.distance(current_norm, precon_norm)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        assert isinstance(current, torch.Tensor) and isinstance(
            goal, torch.Tensor
        ), "Inputs must be torch.Tensor"
        current_norm = self._normalizer.value(current)
        goal_norm = self._normalizer.value(goal)
        value = self._goal_condition.distance(current_norm, goal_norm)
        assert isinstance(value, float), "Distance must be a float"
        assert 0.0 <= value <= 1.0, "Distance must be in [0.0, 1.0]"
        return value

    def evaluate(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        """Evaluate success using injected strategy."""
        assert isinstance(current, torch.Tensor) and isinstance(
            goal, torch.Tensor
        ), "Inputs must be torch.Tensor"
        current_norm = self._eval_normalizer.value(current)
        goal_norm = self._eval_normalizer.value(goal)
        return self._eval_condition.evaluate(current_norm, goal_norm)

    def run_addon(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        addon = self._addons.get(name)
        if addon:
            return addon.run(*args, **kwargs)
        return None
