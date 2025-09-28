from abc import ABC, abstractmethod
import torch
import numpy as np


class ValueConverter(ABC):
    """Abstract base class for state logic strategies."""

    def __init__(self, threshold: float):
        """Base initialization - only threshold is common to all logic types."""
        self.threshold = threshold

    @property
    def name(self) -> str:
        """Returns the name of this logic type."""
        return self.__class__.__name__

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the value method.")

    @abstractmethod
    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute the distance between current state and trajectory point."""
        raise NotImplementedError("Subclasses must implement the tp_distance method.")

    @abstractmethod
    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute the distance between current and goal states."""
        raise NotImplementedError("Subclasses must implement the goal_distance method.")

    @abstractmethod
    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point from start/end states."""
        raise NotImplementedError("Subclasses must implement the make_tp method.")


class DefaultValueConverter(ValueConverter):
    """Default value converter that returns the input as-is."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input value as-is."""
        return x

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute Euclidean distance."""
        return torch.linalg.norm(current - tp).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point if data is consistent."""
        if reversed:
            std = end.std(dim=0)
            if (std < self.threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.threshold).all():
                return start.mean(dim=0)
        return None  # Not consistent enough


class LinearStateLogic(ValueConverter):
    """Base class for linear state logic with normalization - only used by states that need bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        threshold: float = 0.05,
    ):
        super().__init__(threshold)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.relative_threshold = threshold * (upper_bound - lower_bound)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a value x to the range [0, 1] based on bounds."""
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        cx = torch.clamp(x, self.lower_bound, self.upper_bound)
        return self.normalize(cx)


class DiscreteStateLogic(ValueConverter):
    """Base class for discrete state logic - no bounds needed."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input value as-is for discrete states."""
        return x


class EulerAngleLogic(LinearStateLogic):
    """Logic for Euler angle states (3D rotations) - needs bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        threshold: float = 0.05,
    ):
        # Validate that bounds are 3D before calling super
        assert lower_bound.shape == upper_bound.shape == (3,), "Euler angles must be 3D"
        super().__init__(lower_bound, upper_bound, threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute normalized Euclidean distance for Euler angles."""
        nx = self.normalize(current)
        ny = self.normalize(tp)
        return torch.linalg.norm(nx - ny).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point for Euler angles."""
        if not tapas_selection:
            return None  # Tapas didn't select this state
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)


class AxisAngleLogic(LinearStateLogic):
    """Logic for Axis-Angle representations (3D rotations) - needs bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        threshold: float = 0.05,
    ):
        # Validate that bounds are 3D before calling super
        assert lower_bound.shape == upper_bound.shape == (3,), "Axis angles must be 3D"
        super().__init__(lower_bound, upper_bound, threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute normalized distance for axis-angle representation."""
        nx = self.normalize(current)
        ny = self.normalize(tp)
        return torch.linalg.norm(nx - ny).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point for axis angles."""
        if not tapas_selection:
            return None
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)


class QuaternionLogic(ValueConverter):
    """Logic for quaternion states (4D rotations) - no bounds needed."""

    def __init__(self, threshold: float = 0.05):
        # Quaternions don't need bounds - they're always unit quaternions
        super().__init__(threshold)

    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion and ensure positive w component."""
        x = x / torch.linalg.norm(x)
        if x[3] < 0:
            return -x
        return x

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.normalize_quat(x)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute quaternion distance using angular metric."""
        nx = self.normalize_quat(current)
        ny = self.normalize_quat(tp)
        dot = torch.clamp(torch.abs(torch.dot(nx, ny)), 0.0, 1.0)
        return 2.0 * torch.arccos(dot).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point using quaternion averaging."""
        if not tapas_selection:
            return None
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


class RangeLogic(LinearStateLogic):
    """Logic for scalar range states - needs bounds."""

    def __init__(
        self,
        lower_bound: torch.Tensor,
        upper_bound: torch.Tensor,
        threshold: float = 0.05,
    ):
        # Validate that bounds are scalar before calling super
        assert (
            lower_bound.shape == upper_bound.shape == (1,)
        ), "Range states must be scalar"
        super().__init__(lower_bound, upper_bound, threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute normalized distance for range states."""
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
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point if data is consistent."""
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not consistent enough


class BooleanLogic(DiscreteStateLogic):
    """Logic for boolean states - no bounds needed, just 0/1 values."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)
        # For boolean logic, we can define a relative threshold for consistency checks
        self.relative_threshold = torch.tensor([threshold])

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute boolean distance (0 or 1)."""
        return torch.abs(current - tp).item()

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute distance to goal."""
        return self.tp_distance(current, goal, goal)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point if boolean state is consistent."""
        if reversed:
            std = end.std(dim=0)
            if (std < self.relative_threshold).all():
                return end.mean(dim=0)
        else:
            std = start.std(dim=0)
            if (std < self.relative_threshold).all():
                return start.mean(dim=0)
        return None  # Not consistent enough


class FlipLogic(DiscreteStateLogic):
    """Logic for flip states (inverted distance) - no bounds needed."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)

    def tp_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
        tp: torch.Tensor,
    ) -> float:
        """Compute flipped distance (inverted)."""
        return (tp - torch.abs(current - goal)).item()  # Flipped distance

    def goal_distance(
        self,
        current: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute normal distance to goal."""
        return torch.abs(current - goal).item()

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Generate trajectory point if flip occurred."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state detected
        return None


# Logic factory mapping
LOGIC_REGISTRY = {
    "Euler": EulerAngleLogic,
    "Axis": AxisAngleLogic,
    "Quat": QuaternionLogic,
    "Range": RangeLogic,
    "Bool": BooleanLogic,
    "Flip": FlipLogic,
}


def create_logic(
    logic_type: str,
    lower_bound: torch.Tensor = None,
    upper_bound: torch.Tensor = None,
    threshold: float = 0.05,
) -> ValueConverter:
    """Factory function to create logic instances with only necessary parameters."""
    if logic_type not in LOGIC_REGISTRY:
        raise ValueError(f"Unknown logic type: {logic_type}")

    logic_class = LOGIC_REGISTRY[logic_type]

    # Logic types that don't need bounds - only threshold
    if logic_type in ["Quat", "Bool", "Flip"]:
        return logic_class(threshold=threshold)

    # Logic types that need bounds - must provide both bounds
    elif logic_type in ["Euler", "Axis", "Range"]:
        if lower_bound is None or upper_bound is None:
            raise ValueError(
                f"Logic type {logic_type} requires lower_bound and upper_bound parameters"
            )
        return logic_class(lower_bound, upper_bound, threshold)

    else:
        raise ValueError(f"Unhandled logic type: {logic_type}")
