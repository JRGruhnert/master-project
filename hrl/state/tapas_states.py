import torch
from hrl.state.logic.tapas_addons import (
    EulerTapasAddons,
    FlipTapasAddons,
    QuatTapasAddons,
    ScalarTapasAddons,
)
from hrl.state.logic.target_condition import (
    BooleanDistanceCondition,
    EulerDistanceCondition,
    FlipDistanceCondition,
    QuaternionDistanceCondition,
    RangeDistanceCondition,
)
from hrl.state.logic.eval_condition import (
    EvalCondition,
)
from hrl.state.logic.normalizer import (
    LinearNormalizer,
    IgnoreNormalizer,
    QuaternionNormalizer,
)
from hrl.state.state import State, TapasState


@State.register_type("Euler")
class EulerState(TapasState):
    def __init__(
        self,
        name,
        id,
        eval_condition: EvalCondition,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Euler",
            normalizer=LinearNormalizer(
                lower_bound=torch.Tensor([-1.0, -1.0, -1.0]),
                upper_bound=torch.Tensor([1.0, 1.0, 1.0]),
            ),
            skill_condition=EulerDistanceCondition(),
            goal_condition=EulerDistanceCondition(),
            eval_condition=eval_condition,
            tapas_addons=EulerTapasAddons(),
        )


@State.register_type("Quat")
class QuatState(TapasState):
    def __init__(
        self,
        name,
        id,
        eval_condition: EvalCondition,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Quat",
            normalizer=QuaternionNormalizer(),
            skill_condition=QuaternionDistanceCondition(),
            goal_condition=QuaternionDistanceCondition(),
            eval_condition=eval_condition,
            tapas_addons=QuatTapasAddons(),
        )


@State.register_type("Range")
class RangeState(TapasState):
    def __init__(
        self,
        name,
        id,
        lower_bound: float,
        upper_bound: float,
        eval_condition: EvalCondition,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Range",
            normalizer=LinearNormalizer(
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
            eval_condition=eval_condition,
            tapas_addons=ScalarTapasAddons(
                lower_bound=torch.Tensor([lower_bound]),
                upper_bound=torch.Tensor([upper_bound]),
            ),
        )


@State.register_type("Bool")
class BoolState(TapasState):
    def __init__(
        self,
        name,
        id,
        eval_condition: EvalCondition,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Bool",
            normalizer=IgnoreNormalizer(),
            skill_condition=BooleanDistanceCondition(),
            goal_condition=BooleanDistanceCondition(),
            eval_condition=eval_condition,
            tapas_addons=ScalarTapasAddons(
                lower_bound=torch.Tensor([0.0]),
                upper_bound=torch.Tensor([1.0]),
            ),
        )


@State.register_type("Flip")
class FlipState(TapasState):
    def __init__(
        self,
        name,
        id,
        eval_condition: EvalCondition,
    ):
        super().__init__(
            name=name,
            id=id,
            type_str="Flip",
            normalizer=IgnoreNormalizer(),
            skill_condition=FlipDistanceCondition(),
            goal_condition=BooleanDistanceCondition(),
            eval_condition=eval_condition,
            tapas_addons=FlipTapasAddons(),
        )
