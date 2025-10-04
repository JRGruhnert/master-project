import torch
from hrl.common.logic.target_condition import (
    BooleanDistanceCondition,
    EulerDistanceCondition,
    FlipDistanceCondition,
    QuaternionDistanceCondition,
    RangeDistanceCondition,
)
from hrl.common.tapas_state import (
    BoolState,
    EulerState,
    FlipState,
    QuatState,
    RangeState,
)
from hrl.common.logic.eval_condition import (
    AreaEvalCondition,
    PreciseEvalCondition,
    IgnoreEvalCondition,
)

_S = {
    "ee_position": EulerState(
        name="ee_position",
        id=0,
        eval_condition=PreciseEvalCondition(
            target_condition=EulerDistanceCondition(),
        ),
    ),
    "base__slide_position": EulerState(
        name="base__slide_position",
        id=1,
        eval_condition=PreciseEvalCondition(
            target_condition=EulerDistanceCondition(),
        ),
    ),
    "base__drawer_position": EulerState(
        name="base__drawer_position",
        id=2,
        eval_condition=PreciseEvalCondition(
            target_condition=EulerDistanceCondition(),
        ),
    ),
    "base__button_position": EulerState(
        name="base__button_position",
        id=3,
        eval_condition=PreciseEvalCondition(
            target_condition=EulerDistanceCondition(),
        ),
    ),
    "led_position": EulerState(
        name="led_position",
        id=4,
        eval_condition=PreciseEvalCondition(
            target_condition=EulerDistanceCondition(),
        ),
    ),
    "block_red_position": EulerState(
        name="block_red_position",
        id=14,
        eval_condition=AreaEvalCondition(
            surfaces={
                "table": torch.Tensor([[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]]),
                "drawer_open": torch.Tensor([[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]]),
                "drawer_closed": torch.Tensor(
                    [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]]
                ),
            },
            lower_bound=torch.Tensor([-1.0, -1.0, -1.0]),
            upper_bound=torch.Tensor([1.0, 1.0, 1.0]),
        ),
    ),
    "block_blue_position": EulerState(
        name="block_blue_position",
        id=15,
        eval_condition=AreaEvalCondition(
            surfaces={
                "table": torch.Tensor([[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]]),
                "drawer_open": torch.Tensor([[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]]),
                "drawer_closed": torch.Tensor(
                    [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]]
                ),
            },
            lower_bound=torch.Tensor([-1.0, -1.0, -1.0]),
            upper_bound=torch.Tensor([1.0, 1.0, 1.0]),
        ),
    ),
    "block_pink_position": EulerState(
        name="block_pink_position",
        id=16,
        eval_condition=AreaEvalCondition(
            surfaces={
                "table": torch.Tensor([[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]]),
                "drawer_open": torch.Tensor([[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]]),
                "drawer_closed": torch.Tensor(
                    [[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]]
                ),
            },
            lower_bound=torch.Tensor([-1.0, -1.0, -1.0]),
            upper_bound=torch.Tensor([1.0, 1.0, 1.0]),
        ),
    ),
    "ee_rotation": QuatState(
        name="ee_rotation",
        id=5,
        eval_condition=PreciseEvalCondition(
            target_condition=QuaternionDistanceCondition()
        ),
    ),
    "base__slide_rotation": QuatState(
        name="base__slide_rotation",
        id=6,
        eval_condition=PreciseEvalCondition(
            target_condition=QuaternionDistanceCondition()
        ),
    ),
    "base__drawer_rotation": QuatState(
        name="base__drawer_rotation",
        id=7,
        eval_condition=PreciseEvalCondition(
            target_condition=QuaternionDistanceCondition()
        ),
    ),
    "base__button_rotation": QuatState(
        name="base__button_rotation",
        id=8,
        eval_condition=PreciseEvalCondition(
            target_condition=QuaternionDistanceCondition()
        ),
    ),
    "led_rotation": QuatState(
        name="led_rotation",
        id=9,
        eval_condition=PreciseEvalCondition(
            target_condition=QuaternionDistanceCondition()
        ),
    ),
    "block_red_rotation": QuatState(
        name="block_red_rotation",
        id=17,
        eval_condition=IgnoreEvalCondition(),
    ),
    "block_blue_rotation": QuatState(
        name="block_blue_rotation",
        id=18,
        eval_condition=IgnoreEvalCondition(),
    ),
    "block_pink_rotation": QuatState(
        name="block_pink_rotation",
        id=19,
        eval_condition=IgnoreEvalCondition(),
    ),
    "ee_scalar": BoolState(
        name="ee_scalar",
        id=10,
        eval_condition=PreciseEvalCondition(
            target_condition=BooleanDistanceCondition()
        ),
    ),
    "base__slide_scalar": RangeState(
        name="base__slide_scalar",
        id=11,
        lower_bound=0.0,
        upper_bound=0.28,
        eval_condition=PreciseEvalCondition(
            target_condition=RangeDistanceCondition(
                lower_bound=torch.Tensor([0.0]),
                upper_bound=torch.Tensor([0.28]),
            ),
        ),
    ),
    "base__drawer_scalar": RangeState(
        name="base__drawer_scalar",
        id=12,
        lower_bound=torch.Tensor([0.0]),
        upper_bound=torch.Tensor([0.22]),
        eval_condition=PreciseEvalCondition(
            target_condition=RangeDistanceCondition(
                lower_bound=torch.Tensor([0.0]),
                upper_bound=torch.Tensor([0.22]),
            ),
        ),
    ),
    "base__button_scalar": FlipState(
        name="base__button_scalar",
        id=13,
        eval_condition=PreciseEvalCondition(
            target_condition=BooleanDistanceCondition(),
        ),
    ),
    "block_red_scalar": BoolState(
        name="block_red_scalar",
        id=20,
        eval_condition=PreciseEvalCondition(
            target_condition=BooleanDistanceCondition(),
        ),
    ),
    "block_blue_scalar": BoolState(
        name="block_blue_scalar",
        id=21,
        eval_condition=PreciseEvalCondition(
            target_condition=BooleanDistanceCondition(),
        ),
    ),
    "block_pink_scalar": BoolState(
        name="block_pink_scalar",
        id=22,
        eval_condition=PreciseEvalCondition(
            target_condition=BooleanDistanceCondition(),
        ),
    ),
}

STATES_BY_TAG = {
    "Debug": [
        _S["ee_position"],
        _S["base__slide_position"],
        _S["base__drawer_position"],
        _S["base__button_position"],
        _S["led_position"],
        _S["block_red_position"],
        _S["block_blue_position"],
        _S["block_pink_position"],
        _S["ee_rotation"],
        _S["base__slide_rotation"],
        _S["base__drawer_rotation"],
        _S["base__button_rotation"],
        _S["led_rotation"],
        _S["block_red_rotation"],
        _S["block_blue_rotation"],
        _S["block_pink_rotation"],
        _S["ee_scalar"],
        _S["base__slide_scalar"],
        _S["base__drawer_scalar"],
        _S["base__button_scalar"],
        _S["block_red_scalar"],
        _S["block_blue_scalar"],
        _S["block_pink_scalar"],
    ],
    "Normal": [
        _S["ee_position"],
        _S["base__slide_position"],
        _S["base__drawer_position"],
        _S["base__button_position"],
        _S["led_position"],
        _S["block_red_position"],
        _S["block_blue_position"],
        _S["block_pink_position"],
        _S["ee_rotation"],
        _S["base__slide_rotation"],
        _S["base__drawer_rotation"],
        _S["base__button_rotation"],
        _S["led_rotation"],
        _S["block_red_rotation"],
        _S["block_blue_rotation"],
        _S["block_pink_rotation"],
        _S["ee_scalar"],
        _S["base__slide_scalar"],
        _S["base__drawer_scalar"],
        _S["base__button_scalar"],
        _S["block_red_scalar"],
        _S["block_blue_scalar"],
        _S["block_pink_scalar"],
    ],
    "Minimal": [
        _S["ee_position"],
        _S["base__slide_position"],
        _S["base__drawer_position"],
        _S["base__button_position"],
        _S["led_position"],
        _S["ee_rotation"],
        _S["base__slide_rotation"],
        _S["base__drawer_rotation"],
        _S["base__button_rotation"],
        _S["led_rotation"],
        _S["ee_scalar"],
        _S["base__slide_scalar"],
        _S["base__drawer_scalar"],
        _S["base__button_scalar"],
    ],
}
