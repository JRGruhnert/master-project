from abc import ABC, abstractmethod
import torch

from hrl.state.logic.mixin import QuaternionMixin, RelThresholdMixin


class ValueConverter(ABC):
    """Abstract base class for state logic strategies."""

    @abstractmethod
    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the processed value of the state as a tensor."""
        raise NotImplementedError("Subclasses must implement the value method.")


class DefaultValueConverter(ValueConverter):
    """Default value converter that returns the input as-is."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input value as-is."""
        return x  # No processing


class NormalizeValueConverter(ValueConverter, RelThresholdMixin):
    """Default value converter that returns the input as-is."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp and normalize the input value."""
        cx = torch.clamp(x, self.lower_bound, self.upper_bound)
        return self._normalize(cx)


class QuaternionValueConverter(ValueConverter, QuaternionMixin):
    """Value converter for quaternion states - no bounds needed."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the quaternion."""
        return self._normalize_quat(x)
