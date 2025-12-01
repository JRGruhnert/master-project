import pytest
import torch
from src.logic.eval_condition import AreaEvalCondition, PreciseEvalCondition
from src.logic.distance_condition import (
    EulerDistanceCondition,
    QuaternionDistanceCondition,
    RangeDistanceCondition,
    BooleanDistanceCondition,
    FlipDistanceCondition,
)

from src.states.calvin import (
    CalvinState,
    BoolState,
    EulerState,
    FlipState,
    QuatState,
    RangeState,
)

euler_precise = EulerState(
    "test",
    0,
    eval_condition=PreciseEvalCondition(
        condition=EulerDistanceCondition(),
    ),
)

euler_area = EulerState(
    "test",
    0,
    eval_condition=AreaEvalCondition(
        surfaces={
            "table": torch.Tensor([[0.0, -0.15, 0.46], [0.30, -0.03, 0.46]]),
            "drawer_open": torch.Tensor([[0.04, -0.35, 0.38], [0.30, -0.21, 0.38]]),
            "drawer_closed": torch.Tensor([[0.04, -0.16, 0.38], [0.30, -0.03, 0.38]]),
        },
        lower_bound=torch.Tensor([-1.0, -1.0, -1.0]),
        upper_bound=torch.Tensor([1.0, 1.0, 1.0]),
    ),
)

quat_state = QuatState(
    "test",
    0,
    eval_condition=PreciseEvalCondition(
        condition=QuaternionDistanceCondition(),
    ),
)

range_state = RangeState(
    "test",
    0,
    lower_bound=torch.tensor([0.0]),
    upper_bound=torch.tensor([1.0]),
    eval_condition=PreciseEvalCondition(
        condition=RangeDistanceCondition(
            lower_bound=torch.tensor([0.0]), upper_bound=torch.tensor([1.0])
        ),
    ),
)

bool_state = BoolState(
    "test",
    0,
    eval_condition=PreciseEvalCondition(
        condition=BooleanDistanceCondition(),
    ),
)

flip_state = FlipState(
    "test",
    0,
    eval_condition=PreciseEvalCondition(
        condition=BooleanDistanceCondition(),
    ),
)


def tensor_close(actual, expected, atol=1e-6):
    """Helper function to compare tensor/float values safely."""
    # Convert both to tensors with consistent dtype
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual, dtype=torch.float32)
    else:
        actual = actual.float()

    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=torch.float32)
    else:
        expected = expected.float()

    return torch.isclose(actual, expected, atol=atol)


class TestTapasState:
    @pytest.mark.parametrize(
        "state,current,expected",
        [
            # Test the value() method - this should return normalized values
            # Note: You may need to adjust expected values based on actual normalization
            (euler_precise, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),  # Middle of range
            (euler_precise, [-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]),  # Lower bound
            (euler_precise, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),  # Upper bound
            (quat_state, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]),  # Identity quat
            (range_state, [0.0], [0.0]),  # Lower bound
            (range_state, [0.5], [0.5]),  # Middle
            (range_state, [1.0], [1.0]),  # Upper bound
            (bool_state, [0.0], [0.0]),  # False
            (bool_state, [1.0], [1.0]),  # True
        ],
    )
    def test_value(
        self,
        state: CalvinState,
        current: list[float],
        expected: list[float],
    ):
        """Test state value normalization."""
        current_tensor = torch.tensor(current, dtype=torch.float32)
        expected_tensor = torch.tensor(expected, dtype=torch.float32)

        result = state.normalize(current_tensor)

        # Handle multi-element vs single element results
        if result.numel() > 1:
            assert torch.allclose(result, expected_tensor, atol=1e-6)
        else:
            assert tensor_close(result, expected_tensor, atol=1e-6)

    @pytest.mark.parametrize(
        "state, current, goal, expected_approx",
        [
            # Test distance_to_goal method
            (euler_precise, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
            (
                euler_precise,
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
                1.73,
            ),  # Approx sqrt(3)
            (quat_state, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], 0.0),
            (range_state, [0.0], [0.0], 0.0),
            (range_state, [0.0], [1.0], 1.0),
            (bool_state, [0.0], [0.0], 0.0),
            (bool_state, [0.0], [1.0], 1.0),
            (flip_state, [0.0], [0.0], 0.0),
            (flip_state, [1.0], [0.0], 1.0),
        ],
    )
    def test_goal_distance(
        self,
        state: CalvinState,
        current: list[float],
        goal: list[float],
        expected_approx: float,
    ):
        """Test distance to goal calculations."""
        current_tensor = torch.tensor(current, dtype=torch.float32)
        goal_tensor = torch.tensor(goal, dtype=torch.float32)

        distance = state.distance_to_goal(current_tensor, goal_tensor)

        # Convert to float if it's a tensor
        if isinstance(distance, torch.Tensor):
            distance = distance.item()

        assert (
            abs(distance - expected_approx) < 0.1
        ), f"Expected ~{expected_approx}, got {distance}"

    @pytest.mark.parametrize(
        "state,current, goal, precon, expected_approx",
        [
            # Test distance_to_skill method
            (euler_precise, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
            (
                quat_state,
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                0.0,
            ),
            (range_state, [0.0], [0.0], [0.0], 0.0),
            (range_state, [0.0], [1.0], [1.0], 1.0),
            (bool_state, [0.0], [0.0], [0.0], 0.0),
            (bool_state, [1.0], [0.0], [0.0], 1.0),
            (flip_state, [0.0], [0.0], [1.0], 1.0),
            (flip_state, [0.0], [1.0], [1.0], 0.0),
        ],
    )
    def test_skill_distance(
        self,
        state: CalvinState,
        current: list[float],
        goal: list[float],
        precon: list[float],
        expected_approx: float,
    ):
        """Test distance to skill calculations."""
        current_tensor = torch.tensor(current, dtype=torch.float32)
        goal_tensor = torch.tensor(goal, dtype=torch.float32)
        precon_tensor = torch.tensor(precon, dtype=torch.float32)

        distance = state.distance_to_skill(current_tensor, goal_tensor, precon_tensor)
        # Convert to float if it's a tensor
        if isinstance(distance, torch.Tensor):
            distance = distance.item()

        assert (
            abs(distance - expected_approx) < 0.1
        ), f"Expected ~{expected_approx}, got {distance}"

    @pytest.mark.parametrize(
        "state,current, goal,expected",
        [
            # Test evaluation method
            (euler_precise, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True),
            (euler_precise, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], False),
            (quat_state, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], True),
            (quat_state, [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], False),
            (range_state, [0.0], [0.0], True),
            (range_state, [0.5], [1.0], False),
            (range_state, [1.0], [1.0], True),
            (bool_state, [0.0], [0.0], True),
            (bool_state, [1.0], [1.0], True),
            (bool_state, [0.0], [1.0], False),
            (flip_state, [0.0], [0.0], True),
            (flip_state, [1.0], [1.0], True),
            (flip_state, [0.0], [1.0], False),
            (flip_state, [1.0], [0.0], False),
        ],
    )
    def test_evaluate(
        self,
        state: CalvinState,
        current: list[float],
        goal: list[float],
        expected: bool,
    ):
        """Test state evaluation (success/failure)."""
        current_tensor = torch.tensor(current, dtype=torch.float32)
        goal_tensor = torch.tensor(goal, dtype=torch.float32)

        result = state.evaluate(current_tensor, goal_tensor)

        # Convert tensor boolean to Python boolean if necessary
        if isinstance(result, torch.Tensor):
            result = result.item()

        assert result == expected, f"Expected {expected}, got {result}"


# Additional test to debug the specific issues
class TestDebugStates:
    """Debug specific state behaviors."""

    def test_euler_state_value_normalization(self):
        """Debug euler state value normalization."""
        current = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        result = euler_precise.normalize(current)
        print(
            f"🔍 Euler value result: {result}, type: {type(result)}, shape: {result.shape}"
        )
        assert isinstance(result, torch.Tensor)

    def test_quat_state_distance(self):
        """Debug quaternion distance calculation."""
        current = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        goal = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        distance = quat_state.distance_to_goal(current, goal)
        print(f"🔍 Quat distance result: {distance}, type: {type(distance)}")

    def test_range_state_distance(self):
        """Debug range state distance calculation."""
        current = torch.tensor([0.5], dtype=torch.float32)
        goal = torch.tensor([0.0], dtype=torch.float32)
        distance = range_state.distance_to_goal(current, goal)
        print(f"🔍 Range distance result: {distance}, type: {type(distance)}")

    def test_flip_state_evaluation(self):
        """Debug flip state evaluation."""
        current = torch.tensor([1.0], dtype=torch.float32)
        goal = torch.tensor([0.0], dtype=torch.float32)
        result = flip_state.evaluate(current, goal)
        print(f"🔍 Flip eval result: {result}, type: {type(result)}")


if __name__ == "__main__":
    # Run the debug tests first to understand the behavior
    pytest.main([__file__ + "::TestDebugStates", "-v", "-s"])

    # Then run the main tests
    pytest.main([__file__ + "::TestTapasState", "-v"])
