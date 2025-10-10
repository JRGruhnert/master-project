from abc import ABC, abstractmethod
import torch


class BaseAddon(ABC):
    """Abstract base class for additional logic components that are only available at runtime."""

    @abstractmethod
    def run(self, *args, **kwargs) -> torch.Tensor | None:
        """Execute the addon logic."""
        raise NotImplementedError("Subclasses must implement the run method.")
