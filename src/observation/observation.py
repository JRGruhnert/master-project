import numpy as np
from tensordict import TensorDict
import torch

singleton_batch_size = torch.Size([1])


class StateValueDict(TensorDict):

    @classmethod
    def from_tensor_dict(
        cls,
        data: dict[str, torch.Tensor],
        batch_size: torch.Size = singleton_batch_size,
    ) -> "StateValueDict":
        """Create BaseObservation from regular dict"""
        batched_data = {
            k: v.unsqueeze(0) if v.dim() == 1 else v for k, v in data.items()
        }
        return cls(batched_data, batch_size=batch_size)

    @classmethod
    def from_numpy_dict(
        cls, data: dict[str, np.ndarray], batch_size: torch.Size = singleton_batch_size
    ) -> "StateValueDict":
        """Create BaseObservation from regular dict"""
        tensor_data = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0)
            for k, v in data.items()
        }
        return cls(tensor_data, batch_size=batch_size)

    def same_fields(self, other: "StateValueDict") -> bool:
        """Check if two StateValueDicts have the same keys"""
        return set(self.keys()) == set(other.keys())  # type: ignore

    def indexed(self, idx: torch.Tensor) -> "StateValueDict":
        """Return a new StateValueDict with the same keys, but indexed tensors"""
        indexed_data = {k: v[idx] for k, v in self.items()}  # type: ignore
        return StateValueDict(indexed_data, batch_size=torch.Size([len(idx)]))

    def add(self, other: "StateValueDict"):
        assert self.same_fields(other), "Cannot append with different fields"
        for k, v in self.items():
            self[k] = torch.cat([v, other[k]], dim=0)
        self.batch_size = torch.Size([self.batch_size[0] + other.batch_size[0]])
