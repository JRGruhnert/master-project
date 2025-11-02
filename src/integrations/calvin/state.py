import torch
from src.integrations.tapas.addon import (
    TapasAddon,
    EulerTapasAddon,
    FlipTapasAddon,
    QuatTapasAddon,
    ScalarTapasAddon,
)
from src.core.logic.distance_condition import (
    DistanceCondition,
    BooleanDistanceCondition,
    EulerDistanceCondition,
    FlipDistanceCondition,
    QuaternionDistanceCondition,
    RangeDistanceCondition,
)
from src.core.logic.eval_condition import (
    AreaEvalCondition,
    EvalCondition,
    IgnoreEvalCondition,
    PreciseEvalCondition,
)
from src.core.logic.value_condition import (
    ValueCondition,
    QuaternionValueNormalizer,
    LinearValueNormalizer,
    IdentityValue,
)
from src.core.state import BaseState


class CalvinState(BaseState):
    def __init__(
        self,
        name: str,
        id: int,
        type_str: str,
        value_condition: ValueCondition,
        skill_condition: DistanceCondition,
        goal_condition: DistanceCondition,
        eval_condition: EvalCondition,
        addon: TapasAddon,
    ):
        super().__init__(
            name,
            id,
            type_str,
            value_condition,
            skill_condition,
            goal_condition,
            eval_condition,
            {"tapas": addon},
        )


class EulerState(CalvinState):

    def __init__(
        self,
        name,
        id,
        type_str,
        ignore: bool = False,
        lower_bound=[-1.0, -1.0, -1.0],
        upper_bound=[1.0, 1.0, 1.0],
        eval_condition: EvalCondition = PreciseEvalCondition(
            condition=EulerDistanceCondition(
                lower_bound=[-1.0, -1.0, -1.0],
                upper_bound=[1.0, 1.0, 1.0],
            ),
        ),
    ):
        super().__init__(
            name=name,
            id=id,
            type_str=type_str,
            value_condition=LinearValueNormalizer(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            skill_condition=EulerDistanceCondition(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            goal_condition=EulerDistanceCondition(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            ),
            eval_condition=IgnoreEvalCondition() if ignore else eval_condition,
            addon=EulerTapasAddon(),
        )


class PreciseEulerState(EulerState):
    def __init__(
        self,
        name,
        id,
        ignore: bool = False,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Euler",
            ignore=ignore,
            eval_condition=PreciseEvalCondition(
                condition=EulerDistanceCondition(
                    lower_bound=[-1.0, -1.0, -1.0],
                    upper_bound=[1.0, 1.0, 1.0],
                ),
            ),
        )


class AreaEulerState(EulerState):
    def __init__(
        self,
        name,
        id,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Euler",
            eval_condition=AreaEvalCondition(
                spawn_surfaces={
                    "table": [[0.0, -0.15, 0.44], [0.30, -0.03, 0.48]],
                    "drawer_open": [[0.04, -0.35, 0.34], [0.30, -0.21, 0.38]],
                    "drawer_closed": [[0.04, -0.16, 0.34], [0.30, -0.03, 0.38]],
                },
                # surfaces={
                #    "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
                #    "drawer_open": [[0.04, -0.35, 0.36], [0.30, -0.21, 0.36]],
                #    "drawer_closed": [[0.04, -0.16, 0.36], [0.30, -0.03, 0.36]],
                # },
                eval_surfaces={
                    "table": [[-0.02, -0.17, 0.44], [0.32, -0.01, 0.54]],
                    "drawer": [[0.02, -0.37, 0.34], [0.32, -0.23, 0.44]],
                },
            ),
        )


class QuatState(CalvinState):
    def __init__(
        self,
        name: str,
        id: int,
        ignore: bool = False,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Quat",
            value_condition=QuaternionValueNormalizer(),
            skill_condition=QuaternionDistanceCondition(),
            goal_condition=QuaternionDistanceCondition(),
            eval_condition=(
                IgnoreEvalCondition()
                if ignore
                else PreciseEvalCondition(condition=QuaternionDistanceCondition())
            ),
            addon=QuatTapasAddon(),
        )


class RangeState(CalvinState):
    def __init__(
        self,
        name,
        id,
        lower_bound: float,
        upper_bound: float,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Range",
            value_condition=LinearValueNormalizer(
                lower_bound=[lower_bound],
                upper_bound=[upper_bound],
            ),
            skill_condition=RangeDistanceCondition(
                lower_bound=[lower_bound],
                upper_bound=[upper_bound],
            ),
            goal_condition=RangeDistanceCondition(
                lower_bound=[lower_bound],
                upper_bound=[upper_bound],
            ),
            eval_condition=PreciseEvalCondition(
                condition=RangeDistanceCondition(
                    lower_bound=[lower_bound],
                    upper_bound=[upper_bound],
                ),
            ),
            addon=ScalarTapasAddon(
                lower_bound=[lower_bound],
                upper_bound=[upper_bound],
            ),
        )


class BoolState(CalvinState):
    def __init__(
        self,
        name,
        id,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Bool",
            value_condition=IdentityValue(),
            skill_condition=BooleanDistanceCondition(),
            goal_condition=BooleanDistanceCondition(),
            eval_condition=PreciseEvalCondition(
                condition=BooleanDistanceCondition(),
            ),
            addon=ScalarTapasAddon(
                lower_bound=[0.0],
                upper_bound=[1.0],
            ),
        )


class FlipState(CalvinState):
    def __init__(
        self,
        name,
        id,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Flip",
            value_condition=IdentityValue(),
            skill_condition=FlipDistanceCondition(),
            goal_condition=BooleanDistanceCondition(),
            eval_condition=PreciseEvalCondition(
                condition=BooleanDistanceCondition(),
            ),
            addon=FlipTapasAddon(),
        )
