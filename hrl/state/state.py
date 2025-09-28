from enum import Enum
import torch
import json
from pathlib import Path

from hrl.state.logic.goal_condition import (
    DefaultGoalCondition,
    GoalCondition,
)
from hrl.state.logic.precon_converter import (
    DefaultPreconConverter,
    PreconConverter,
)
from hrl.state.logic.skill_condition import (
    DefaultSkillCondition,
    SkillCondition,
)
from hrl.state.logic.value_converter import (
    DefaultValueConverter,
    ValueConverter,
)
from hrl.state.logic.success_condition import (
    DefaultSuccessCondition,
    SuccessCondition,
)


class StateSpace(Enum):
    Minimal = "Minimal"
    Normal = "Normal"
    Full = "Full"
    Debug = "Debug"


# class StateType(Enum):
#    Euler_Angle = "Euler"
#    Axis_Angle = "Axis"
#    Quaternion = "Quat"
#    Range = "Range"
#    Boolean = "Bool"
#    Flip = "Flip"  # Special boolean type for flipping the distance


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

    @classmethod
    def _create_state_by_type(
        cls,
        **kwargs,
    ) -> "State":
        """Factory method using registry"""
        if "type" not in kwargs:
            raise ValueError(f"Unknown state type: {kwargs.get('type')}")

        state_class = cls._state_registry[kwargs["type"]]
        return state_class(**kwargs)

    @classmethod
    def from_json_list(cls, state_space: str, relative_path: str) -> list["State"]:
        """Convert a StateSpace to a list of State objects by reading from states.json"""
        # Load the states.json file
        path = Path(relative_path)

        if not path.exists():
            raise FileNotFoundError(f"States JSON file not found at {path}")

        with open(path, "r") as f:
            data: dict = json.load(f)

        # Filter states based on the requested state space
        filtered = []

        for state_key, state_value in data.items():
            # Check if this state belongs to the requested space
            state_space_list = state_value.get("space")
            if state_space_list is None:
                raise ValueError(f"State {state_key} does not have a 'space' defined.")

            if state_space in state_space_list:
                state = cls._create_state_by_type(state_key, **state_value)
                filtered.append(state)

        return filtered

    def __init__(
        self,
        name: str,
        id: int,
        type_str: str,
        value_converter: ValueConverter = DefaultValueConverter(),
        skill_condition: SkillCondition = DefaultSkillCondition(),
        goal_condition: GoalCondition = DefaultGoalCondition(),
        precon_converter: PreconConverter = DefaultPreconConverter(),
        success_condition: SuccessCondition = DefaultSuccessCondition(),
    ):
        self._name = name
        self._id = id
        self._type_str = type_str
        self._value_converter = value_converter
        self._skill_condition = skill_condition
        self._goal_condition = goal_condition
        self._precon_converter = precon_converter
        self._success_condition = success_condition

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
        raise self._value_converter.value(x)

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
        return self._goal_condition.distance(current, goal)

    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor:
        """Returns the mean of the given tensor values."""
        return self._precon_converter.make_tp(start, end, reversed)

    def evaluate(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        """Evaluate success using injected strategy."""
        return self._success_condition.evaluate(obs, goal)
