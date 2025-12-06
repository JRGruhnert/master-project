from abc import ABC, abstractmethod
import torch

from src.logic.mixin import AreaMixin, BoundedMixin, QuaternionMixin


class ValueCondition(ABC):
    """Abstract base class for state logic strategies."""

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the value method.")

    @abstractmethod
    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed input value as a tensor."""
        raise NotImplementedError("Subclasses must implement the make_input method.")


class IdentityValue(ValueCondition):
    """Value converter for discrete states."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return x

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return self.value(x)


class LinearValueNormalizer(ValueCondition, BoundedMixin):
    """Default value converter that returns the input as-is."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        return self.normalize(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        return self.value(x)


class AreaValueNormalizer(ValueCondition, BoundedMixin, AreaMixin):
    """Value converter for area-based states."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Check if the position is within the evaluation areas and normalize."""
        return self.normalize(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Check if the position is within the evaluation areas and normalize."""
        nx = self.value(x)
        area_name = self.check_eval_area(x)
        one_hot_tensor = self.get_one_hot_area_vector(area_name)
        return torch.cat([nx, one_hot_tensor], dim=0)


class QuaternionValueNormalizer(ValueCondition, QuaternionMixin):
    """Value converter for quaternion states - no bounds needed."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.normalize_quat(x)

    def make_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.value(x)
