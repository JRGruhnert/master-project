import torch
from hrl.state.logic.goal_condition import IgnoreGoalCondition
from hrl.state.logic.precon_converter import DefaultPreconConverter
from hrl.state.logic.skill_condition import DefaultSkillCondition
from hrl.state.logic.success_condition import (
    EulerPrecisionSuccessCondition,
    SuccessCondition,
)
from hrl.state.logic.value_converter import (
    DefaultValueConverter,
    NormalizeValueConverter,
    QuaternionValueConverter,
)
from hrl.state.state import State


@State.register_type("Euler")
class EulerState(State):
    def __init__(
        self,
        name,
        id,
        success_condition: SuccessCondition,
    ):
        super().__init__(
            name,
            id,
            "Euler",
            NormalizeValueConverter(),
            DefaultSkillCondition(),
            IgnoreGoalCondition(),
            DefaultPreconConverter(),
            success_condition,
        )


@State.register_type("Quat")
class QuatState(State):
    def __init__(
        self,
        name,
        id,
    ):
        super().__init__(
            name,
            id,
            "Quat",
            QuaternionValueConverter(),
            DefaultSkillCondition(),
            IgnoreGoalCondition(),
            DefaultPreconConverter(),
            EulerPrecisionSuccessCondition(),
        )


@State.register_type("Range")
class RangeState(State):
    def __init__(
        self,
        name,
        id,
        lower_bound: float,
        upper_bound: float,
    ):
        super().__init__(
            name,
            id,
            "Range",
            NormalizeValueConverter(),
            DefaultSkillCondition(),
            IgnoreGoalCondition(),
            DefaultPreconConverter(),
            EulerPrecisionSuccessCondition(
                lower_bound=lower_bound, upper_bound=upper_bound
            ),
        )


@State.register_type("Bool")
class BoolState(State):
    def __init__(
        self,
        name,
        id,
    ):
        super().__init__(
            name,
            id,
            "Bool",
            DefaultValueConverter(),
            DefaultSkillCondition(),
            IgnoreGoalCondition(),
            DefaultPreconConverter(),
            EulerPrecisionSuccessCondition(),
        )


@State.register_type("Flip")
class FlipState(State):
    def __init__(
        self,
        name,
        id,
    ):
        super().__init__(
            name,
            id,
            "Flip",
            DefaultValueConverter(),
            DefaultSkillCondition(),
            IgnoreGoalCondition(),
            DefaultPreconConverter(),
            EulerPrecisionSuccessCondition(),
        )
