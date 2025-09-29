from abc import ABC, abstractmethod
import torch
import numpy as np

from hrl.state.logic.mixin import QuaternionMixin, RelThresholdMixin


class PreconConverter(ABC):
    """Abstract base class for state logic strategies."""

    @abstractmethod
    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        """Generate trajectory point from start/end states."""
        raise NotImplementedError("Subclasses must implement the make_tp method.")


class DefaultPreconConverter(PreconConverter):
    """Default no-op logic - returns input as-is."""

    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)


class ScalarPreconConverter(PreconConverter, RelThresholdMixin):
    """Scalar precondition converter."""

    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
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


class FlipPreconConverter(PreconConverter):
    """Flip precondition converter for boolean states."""

    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state
        return None


class QuaternionPreconConverter(PreconConverter, QuaternionMixin):
    """Quaternion precondition converter."""

    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        if reversed:
            return self._quaternion_mean(end)
        return self._quaternion_mean(start)


class EulerPreconConverter(PreconConverter):
    """Euler angle precondition converter."""

    def retrieve_precon(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)
