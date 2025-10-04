from abc import ABC, abstractmethod
import torch

from hrl.common.logic.mixin import QuaternionMixin, RelThresholdMixin


class TapasAddons(ABC):
    """Abstract base class for state logic strategies."""

    @abstractmethod
    def make_tps(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        """Generate trajectory point from start/end states."""
        raise NotImplementedError("Subclasses must implement the make_tp method.")


class ScalarTapasAddons(TapasAddons, RelThresholdMixin):
    """Scalar precondition converter."""

    def make_tps(
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


class FlipTapasAddons(TapasAddons):
    """Flip precondition converter for boolean states."""

    def make_tps(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state
        return None


class QuatTapasAddons(TapasAddons, QuaternionMixin):
    """Quaternion precondition converter."""

    def make_tps(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        if reversed:
            return self._quaternion_mean(end)
        return self._quaternion_mean(start)


class EulerTapasAddons(TapasAddons):
    """Euler angle precondition converter."""

    def make_tps(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
    ) -> torch.Tensor | None:
        if reversed:
            return end.mean(dim=0)
        return start.mean(dim=0)
