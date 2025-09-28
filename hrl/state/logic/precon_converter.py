from abc import ABC, abstractmethod
import torch
import numpy as np


class PreconConverter(ABC):
    """Abstract base class for state logic strategies."""

    def __init__(self, threshold: float):
        """Base initialization - only threshold is common to all logic types."""
        self.threshold = threshold

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


class DefaultPreconConverter(PreconConverter):
    """Default no-op logic - returns input as-is."""

    def __init__(self, threshold: float = 0.05):
        super().__init__(threshold)

    def make_tp(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        tapas_selection: bool = True,
    ) -> torch.Tensor | None:
        """Return None - no trajectory point generation."""
        return None
