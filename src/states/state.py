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
    ):
        self.name = name
        self.id = id
        self.type = type_str
        self.size = size
        self._normalizer = normalizer
        self._skill_condition = skill_condition
        self._goal_condition = goal_condition
        self._eval_condition = eval_condition
        self._addons = addons

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the value of the state as a tensor."""
        return self._normalizer.value(x)

    def distance_to_skill(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        precon: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self._skill_condition.distance(current, goal, precon)

    def distance_to_goal(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self._goal_condition.distance(current, goal, goal)

    def evaluate(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        """Evaluate success using injected strategy."""
        return self._eval_condition.evaluate(current, goal)

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
