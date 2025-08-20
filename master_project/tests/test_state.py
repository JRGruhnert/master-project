import pytest
import torch
import numpy as np

from tapas_gmm.master_project.state import (
    StateType,
    State,
    EulerState,
    QuaternionState,
    RangeState,
    BooleanState,
    FlipState,
)

# Example test cases for each state type
STATE_TEST_CASES = [
    # EulerState: normalized value and Euclidean distance
    (
        EulerState(
            "euler",
            StateType.Euler_Angle,
            None,
            torch.tensor([-1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0]),
        ),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.linalg.norm(torch.tensor([0.5, 0.5, 0.5])),
    ),
    # QuaternionState: normalized quaternion and angular distance
    (
        QuaternionState(
            "quat",
            StateType.Quaternion,
            None,
            torch.tensor([-1.0, -1.0, -1.0, -1.0]),
            torch.tensor([1.0, 1.0, 1.0, 1.0]),
        ),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        torch.tensor([0.0, 0.0, 0.0, 1.0]),
        torch.tensor(0.0),
    ),
    # RangeState: normalized value and absolute difference
    (
        RangeState(
            "range", StateType.Range, None, torch.tensor([0.0]), torch.tensor([10.0])
        ),
        torch.tensor([5.0]),
        torch.tensor([10.0]),
        torch.tensor([0.5]),
        torch.tensor(0.5),
    ),
    # BooleanState: value and absolute difference
    (
        BooleanState(
            "bool", StateType.Boolean, None, torch.tensor([0.0]), torch.tensor([1.0])
        ),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor(1.0),
    ),
    # FlipState: value and flip distance
    (
        FlipState(
            "flip", StateType.Flip, None, torch.tensor([0.0]), torch.tensor([1.0])
        ),
        torch.tensor([1.0]),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        torch.tensor(1.0),
    ),
]


@pytest.mark.parametrize("state, value, goal, exp_value, exp_dist", STATE_TEST_CASES)
def test_state_value_and_distance(
    state: State,
    value: torch.Tensor,
    goal: torch.Tensor,
    exp_value: torch.Tensor,
    exp_dist: torch.Tensor,
):
    # Test value function
    val = state.value(value)
    assert torch.allclose(
        val, exp_value, atol=1e-6
    ), f"{state.name} value failed: {val} != {exp_value}"

    # Test distance_to_goal function
    dist = state.distance_to_goal(value, goal)
    assert torch.allclose(
        dist, torch.tensor(exp_dist), atol=1e-6
    ), f"{state.name} distance failed: {dist} != {exp_dist}"
