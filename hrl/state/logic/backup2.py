from abc import ABC, abstractmethod
import torch
import numpy as np


class StateLogic(ABC):
    """Abstract base class for distance calculation strategies."""

    @abstractmethod
    def value(
        self,
        obs: torch.Tensor,
    ) -> bool:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @abstractmethod
    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute the distance between the current and goal states."""
        raise NotImplementedError(
            "Subclasses must implement the compute_distance method."
        )

    @abstractmethod
    def goal_distance(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Evaluate success condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @abstractmethod
    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor:
        """Returns the mean of the given tensor values."""
        raise NotImplementedError("Subclasses must implement the make_tp method.")


class EuclideanStateLogic(StateLogic):
    """Distance logic based on Euclidean distance."""

    def compute_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        return torch.norm(current - goal).item()

    def evaluate(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        reset: bool = False,
    ) -> bool:
        distance = self.compute_distance(obs, goal)
        # Define a threshold for success (this could be parameterized)
        threshold = 0.05
        return distance <= threshold


class LinearState(State):
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize a value x to the range [0, 1] based on min and max.
        """
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        cx = torch.clamp(x, self.lower_bound, self.upper_bound)
        return self.normalize(cx)


class DiscreteState(State):
    def value(self, x: torch.Tensor) -> torch.Tensor:
        return x


@State.register_type(StateType.Euler_Angle)
class EulerState(LinearState):

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self.normalize(current)
        ny = self.normalize(tp)
        return torch.linalg.norm(nx - ny)

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if not tapas_selection:
            return None  # Tapas didn't select this state
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)


@State.register_type(StateType.Quaternion)
class QuaternionState(State):
    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.linalg.norm(x)
        if x[3] < 0:
            return -x
        return x

    def value(self, x):
        return self.normalize_quat(x)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        nx = self.normalize_quat(current)
        ny = self.normalize_quat(tp)
        dot = torch.clamp(torch.abs(torch.dot(nx, ny)), -1.0, 1.0)
        return 2.0 * torch.arccos(dot)

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if not tapas_selection:
            return None  # Tapas didn't select this state
        if reversed:
            return self.quaternion_mean(end)
        return self.quaternion_mean(start)

    def quaternion_mean(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean quaternion using the eigenvector method.
        quaternions: tensor of shape [N, 4] (x, y, z, w)
        Returns: mean quaternion [4] in (x, y, z, w) format
        """
        # Swap to (w, x, y, z) for computation
        quats = quaternions[:, [3, 0, 1, 2]]
        quats = quats / quats.norm(dim=1, keepdim=True)
        A = quats.t() @ quats
        _, eigenvectors = torch.linalg.eigh(A)
        mean_quat = eigenvectors[:, -1]
        # Ensure positive scalar part
        if mean_quat[0] < 0:
            mean_quat = -mean_quat
        # Swap back to (x, y, z, w)
        mean_quat_xyzw = mean_quat[[1, 2, 3, 0]]
        return self.normalize_quat(mean_quat_xyzw)


@State.register_type(StateType.Range)
class RangeState(LinearState):

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        cx = torch.clamp(current, self.lower_bound, self.upper_bound)
        cy = torch.clamp(tp, self.lower_bound, self.upper_bound)
        nx = self.normalize(cx)
        ny = self.normalize(cy)
        return torch.abs(nx - ny).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not constant enough


@State.register_type(StateType.Boolean)
class BooleanState(DiscreteState):

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        return torch.abs(current - tp).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not constant enough


@State.register_type(StateType.Flip)
class FlipState(DiscreteState):

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return (tp - torch.abs(current - goal)).item()  # Flips distance

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Returns the distance of the state as a tensor."""
        return torch.abs(current - goal).item()

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state
        return None
