from abc import ABC, abstractmethod
import torch

from hrl.common.logic.mixin import BoundedMixin, QuaternionMixin


class ValueCondition(ABC):
    """Abstract base class for state logic strategies."""

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the value method.")


class IdentityValue(ValueCondition):
    """Value converter for discrete states."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return input value as it is."""
        return x


class LinearValueNormalizer(ValueCondition, BoundedMixin):
    """Default value converter that returns the input as-is."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        cx = torch.clamp(x, self.lower_bound, self.upper_bound)
        return self.normalize(cx)


class QuaternionValueNormalizer(ValueCondition, QuaternionMixin):
    """Value converter for quaternion states - no bounds needed."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self.normalize_quat(x)
