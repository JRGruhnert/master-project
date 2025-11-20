import torch

from src.logic.addon import BaseAddon
from src.logic.mixin import QuaternionMixin, RelThresholdMixin


class TapasAddon(BaseAddon):
    """Base class for tapas precondition converters."""

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        """Returns the Taskparameter or None"""
        raise NotImplementedError("Subclasses must implement the run method.")


class ScalarTapasAddon(TapasAddon, RelThresholdMixin):
    """Scalar precondition converter."""

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
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


class FlipTapasAddon(TapasAddon):
    """Flip precondition converter for boolean states."""

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        """Returns the mean of the given tensor values."""
        if (end == (1 - start)).all(dim=0).all():
            return torch.tensor([1.0])  # Flip state
        return None


class QuatTapasAddon(TapasAddon, QuaternionMixin):
    """Quaternion precondition converter."""

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        if selected_by_tapas:
            if reversed:
                return self._quaternion_mean(end)
            return self._quaternion_mean(start)
        return None


class EulerTapasAddon(TapasAddon):
    """Euler angle precondition converter."""

    def run(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        if selected_by_tapas:
            if reversed:
                return end.mean(dim=0)
            return start.mean(dim=0)
        return None  # Not selected by tapas
