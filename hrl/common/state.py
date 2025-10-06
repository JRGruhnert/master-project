import torch

from hrl.common.logic.addon import BaseAddon
from hrl.common.logic.distance_condition import (
    DistanceCondition,
)
from hrl.common.logic.value_condition import (
    ValueCondition,
)
from hrl.common.logic.eval_condition import (
    EvalCondition,
)


class BaseState:

    def __init__(
        self,
        name: str,
        id: int,
        type_str: str,
        normalizer: ValueCondition,
        skill_condition: DistanceCondition,
        goal_condition: DistanceCondition,
        eval_condition: EvalCondition,
        addons: dict[str, BaseAddon],
    ):
        self._name = name
        self._id = id
        self._type_str = type_str
        self._normalizer = normalizer
        self._skill_condition = skill_condition
        self._goal_condition = goal_condition
        self._eval_condition = eval_condition
        self._addons = addons

    @property
    def name(self) -> str:
        """Returns the StateIdent of the state."""
        return self._name

    @property
    def id(self) -> int:
        """Returns the ID of the state."""
        return self._id

    @property
    def type_str(self) -> str:
        """Returns the Type of the state."""
        return self._type_str

    def value(self, x: torch.Tensor) -> torch.Tensor:
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
    ) -> torch.Tensor:
        """Returns the mean of the given tensor values."""
        return self._addons.get(name).run(*args, **kwargs)
