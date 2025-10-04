import torch
from hrl.common.logic.tapas_addons import (
    EulerTapasAddons,
    FlipTapasAddons,
    QuatTapasAddons,
    ScalarTapasAddons,
    TapasAddons,
)
from hrl.common.logic.target_condition import (
    BooleanDistanceCondition,
    EulerDistanceCondition,
    FlipDistanceCondition,
    QuaternionDistanceCondition,
    RangeDistanceCondition,
    TargetCondition,
)
from hrl.common.logic.eval_condition import (
    EvalCondition,
)
from hrl.common.logic.normalizer import (
    LinearNormalizer,
    IgnoreNormalizer,
    Normalizer,
    QuaternionNormalizer,
)
from hrl.common.state import State


class TapasState(State):
    def __init__(
        self,
        name: str,
        id: int,
        type_str: str,
        normalizer: Normalizer,
        skill_condition: TargetCondition,
        goal_condition: TargetCondition,
        eval_condition: EvalCondition,
        tapas_addons: TapasAddons,
    ):
        super().__init__(
            name,
            id,
            type_str,
            normalizer,
            skill_condition,
            goal_condition,
            eval_condition,
        )
        self._tapas_addons = tapas_addons

    def make_additional_tps(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor:
        """Returns the mean of the given tensor values."""
        if not tapas_selection and (
            isinstance(self._tapas_addons, EulerTapasAddons)
            or isinstance(self._tapas_addons, QuatTapasAddons)
        ):  # NOTE: Hack cause tapas does selects them themself
            return None
        return self._tapas_addons.make_tps(start, end, reversed)


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
