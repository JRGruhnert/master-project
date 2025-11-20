from dataclasses import dataclass
import torch

from src.observation.observation import StateValueDict


@dataclass
class BufferConfig:
    batch_size: int = 2048


class Buffer:
    def __init__(self, config: BufferConfig):
        self.config = config

        self.current: list[StateValueDict] = []
        self.goal: list[StateValueDict] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.success: list[bool] = []
        self.values: list[torch.Tensor] = []
        self.terminals: list[bool] = []

    def clear(self):
        self.current.clear()
        self.goal.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.success.clear()
        self.values.clear()
        self.terminals.clear()

    def health(self):
        lengths = [
            len(self.current),
            len(self.goal),
            len(self.actions),
            len(self.logprobs),
            len(self.rewards),
            len(self.success),
            len(self.values),
            len(self.terminals),
        ]
        return all(l == lengths[0] for l in lengths)

    def save(self, path: str, epoch: int) -> float:
        """Saves the current batch to a file and returns the success rate of the batch."""
        assert self.health(), "Buffer lengths are inconsistent!"

        file_path = path + f"stats_epoch_{epoch}.pt"
        data = {
            "actions": torch.stack(self.actions),
            "logprobs": torch.tensor(self.logprobs),
            "values": torch.tensor(self.values),
            "rewards": torch.tensor(self.rewards),
            "success": torch.tensor(self.success),
            "terminals": torch.tensor(self.terminals),
        }
        torch.save(data, file_path)
        return self.success_rate()

    def success_rate(self) -> float:
        return self.success.count(True) / max(1, self.terminals.count(True))

    def act_values(
        self,
        current: StateValueDict,
        goal: StateValueDict,
        action: torch.Tensor,
        action_logprob: torch.Tensor,
        state_val: torch.Tensor,
    ):
        self.current.append(current)
        self.goal.append(goal)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.values.append(state_val)

    def act_values_tree(
        self,
        current: StateValueDict,
        goal: StateValueDict,
        action: int,
    ):
        self.current.append(current)
        self.goal.append(goal)
        self.actions.append(torch.tensor(action))
        self.logprobs.append(torch.tensor(0.0))
        self.values.append(torch.tensor(0.0))

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        self.rewards.append(reward)
        self.success.append(success)
        self.terminals.append(terminal)
        assert self.health(), "Buffer lengths are inconsistent!"
        return len(self.actions) == self.config.batch_size

    def metrics(self) -> dict[str, float]:
        assert self.health(), "Buffer lengths are inconsistent!"
        return {
            "total_reward": sum(self.rewards),
            "average_reward": sum(self.rewards) / self.config.batch_size,
            "average_length": self.config.batch_size
            / max(1, self.terminals.count(True)),
            "success_rate": self.success_rate(),
        }
