from abc import abstractmethod
from dataclasses import dataclass
import torch
from pathlib import Path
from torch_geometric.data import HeteroData

from collections import deque
from heca.misc.base import Configurable
from heca.misc import logger


@dataclass(slots=True)
class BufferData:
    data: HeteroData
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor
    reward: float = 0.0
    terminal: bool = False
    truncated: bool = False


class Buffer(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        capacity: int
        gamma: float

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.queue: deque[BufferData] = deque(maxlen=cfg.capacity)
        self.step_pointer = 0
        self.tag_capacity = self.cfg.capacity

    def reset(self):
        self.step_pointer = 0

    def add(self, data: BufferData) -> bool:
        self.queue.append(data)
        self.step_pointer += 1
        return self.full

    def stats(self) -> dict[str, float]:
        n_truncs = sum(self.truncates)
        n_terms = sum(self.terminals)

        # Count episodes where last step has terminal=True and reward > step penalty
        n_success = sum(
            1 for d in self.queue if d.terminal and d.reward > 0.0  # or > 0
        )

        total = n_terms + n_truncs
        return {
            "stats/n_episodes": total,
            "stats/success_rate": n_success / total if total > 0 else 0.0,
            "stats/mean_length": 2048 / total if total > 0 else 0.0,
        }

    @abstractmethod
    def compute_advantages(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def full(self) -> bool:
        return self.cfg.capacity <= self.step_pointer

    @property
    def data(self) -> list[HeteroData]:
        return [d.data for d in self.queue]

    @property
    def actions(self) -> torch.Tensor:
        return torch.stack([d.action for d in self.queue])

    @property
    def logprobs(self) -> torch.Tensor:
        return torch.stack([d.logprob for d in self.queue])

    @property
    def values(self) -> torch.Tensor:
        return torch.stack([d.value for d in self.queue])

    @property
    def rewards(self) -> list[float]:
        return [d.reward for d in self.queue]

    @property
    def terminals(self) -> list[bool]:
        return [d.terminal for d in self.queue]

    @property
    def truncates(self) -> list[bool]:
        return [d.truncated for d in self.queue]

    def save(self, path: Path, label: str):
        logger.info(f"Saving buffer '{label}'")
        serialized = {
            "items": list(self.queue),
        }
        torch.save(serialized, path / f"buffer_{label}.pt")

    def load(self, path: Path, label: str):
        logger.info(f"Loading buffer '{label}'")
        serialized = torch.load(path / f"buffer_{label}.pt")
        self.queue = deque(serialized["items"], maxlen=self.cfg.capacity)
