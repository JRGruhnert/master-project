import torch

from hrl.common.logic.target_condition import (
    TargetCondition,
)
from hrl.common.logic.normalizer import (
    Normalizer,
)
from hrl.common.logic.eval_condition import (
    EvalCondition,
)


class State:
    _state_registry = {}

    @classmethod
    def register_type(cls, state_type: str):
        """Decorator to register state types"""

        def decorator(state_class):
            if state_type in cls._state_registry:
                raise ValueError(f"State type {state_type} is already registered.")
            cls._state_registry[state_type] = state_class
            return state_class

        return decorator

    def __init__(
        self,
        name: str,
        id: int,
        type_str: str,
        normalizer: Normalizer,
        skill_condition: TargetCondition,
        goal_condition: TargetCondition,
        eval_condition: EvalCondition,
    ):
        self._name = name
        self._id = id
        self._type_str = type_str
        self._normalizer = normalizer
        self._skill_condition = skill_condition
        self._goal_condition = goal_condition
        self._eval_condition = eval_condition

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
