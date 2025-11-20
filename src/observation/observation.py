import numpy as np
from tensordict import TensorDict
import torch


class StateValueDict(TensorDict):

    @classmethod
    def from_tensor_dict(cls, data: dict[str, torch.Tensor]) -> "StateValueDict":
        """Create BaseObservation from regular dict"""
        return cls(data, batch_size=[])

    @classmethod
    def from_numpy_dict(cls, data: dict[str, np.ndarray]) -> "StateValueDict":
        """Create BaseObservation from regular dict"""
        tensor_data = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
        return cls(tensor_data, batch_size=[])

    def same_fields(self, other: "StateValueDict") -> bool:
        """Check if two StateValueDicts have the same keys"""
        return set(self.keys()) == set(other.keys())  # type: ignore
