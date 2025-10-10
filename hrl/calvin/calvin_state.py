import torch
from hrl.tapas.tapas_addon import (
    EulerTapasAddon,
    FlipTapasAddon,
    QuatTapasAddon,
    ScalarTapasAddon,
)
from hrl.common.logic.distance_condition import (
    DistanceCondition,
    BooleanDistanceCondition,
    EulerDistanceCondition,
    FlipDistanceCondition,
    QuaternionDistanceCondition,
    RangeDistanceCondition,
)
from hrl.common.logic.eval_condition import (
    AreaEvalCondition,
    EvalCondition,
    IgnoreEvalCondition,
    PreciseEvalCondition,
)
from hrl.common.logic.value_condition import (
    ValueCondition,
    QuaternionValueNormalizer,
    LinearValueNormalizer,
    IdentityValue,
)
from hrl.common.state import BaseState
from hrl.tapas.tapas_addon import TapasAddon


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

    def run_addon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor:
        """Returns the mean of the given tensor values."""
        return super().run_addon(
            "tapas",
            start=start,
            end=end,
            reversed=reversed,
            selected_by_tapas=selected_by_tapas,
        )


class EulerState(CalvinState):
    def __init__(
        self,
        name,
        id,
        ignore: bool = False,
        eval_condition: EvalCondition = PreciseEvalCondition(
            distance_condition=EulerDistanceCondition(),
        ),
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Euler",
            value_condition=LinearValueNormalizer(
                lower_bound=torch.Tensor([-1.0, -1.0, -1.0]),
                upper_bound=torch.Tensor([1.0, 1.0, 1.0]),
            ),
            skill_condition=EulerDistanceCondition(),
            goal_condition=EulerDistanceCondition(),
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
            ignore=ignore,
            eval_condition=PreciseEvalCondition(
                distance_condition=EulerDistanceCondition(),
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
            eval_condition=AreaEvalCondition(
                surfaces={
                    "table": [[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]],
                    "drawer_open": [[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]],
                    "drawer_closed": [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]],
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
                else PreciseEvalCondition(
                    distance_condition=QuaternionDistanceCondition()
                )
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
                lower_bound=torch.Tensor([lower_bound]),
                upper_bound=torch.Tensor([upper_bound]),
            ),
            skill_condition=RangeDistanceCondition(
                lower_bound=torch.Tensor([lower_bound]),
                upper_bound=torch.Tensor([upper_bound]),
            ),
            goal_condition=RangeDistanceCondition(
                lower_bound=torch.Tensor([lower_bound]),
                upper_bound=torch.Tensor([upper_bound]),
            ),
            eval_condition=PreciseEvalCondition(
                distance_condition=RangeDistanceCondition(
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                ),
            ),
            addon=ScalarTapasAddon(
                lower_bound=torch.Tensor([lower_bound]),
                upper_bound=torch.Tensor([upper_bound]),
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
                distance_condition=BooleanDistanceCondition(),
            ),
            addon=ScalarTapasAddon(
                lower_bound=torch.Tensor([0.0]),
                upper_bound=torch.Tensor([1.0]),
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
                distance_condition=BooleanDistanceCondition(),
            ),
            addon=FlipTapasAddon(),
        )
