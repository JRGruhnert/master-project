from functools import cached_property

from tensordict import TensorDict
import torch


class BaseObservation(TensorDict):

    @cached_property
    def top_level_observation(self) -> dict[str, torch.Tensor]:
        """Returns the top-level observation as a standard dictionary."""
        return dict(self)
