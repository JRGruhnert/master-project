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
        return sum(self.success) / len(self.success) if self.success else 0

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
        # Calculate episode statistics
        episode_rewards = []
        episode_lengths_batch = []
        episode_success = []

        current_episode_reward = 0
        current_episode_length = 0

        for i, reward in enumerate(self.rewards):
            current_episode_reward += reward
            current_episode_length += 1
            current_episode_success = int(self.success[i])

            if i == len(self.rewards) - 1 or self.terminals[i]:
                episode_rewards.append(current_episode_reward)
                episode_lengths_batch.append(current_episode_length)
                episode_success.append(current_episode_success)
                current_episode_reward = 0
                current_episode_length = 0

        return {
            "total_reward": sum(episode_rewards),
            "average_length": (
                sum(episode_lengths_batch) / len(episode_lengths_batch)
                if episode_lengths_batch
                else 0
            ),
            "success_rate": (
                sum(episode_success) / len(episode_success) if episode_success else 0
            ),
        }
