import torch

from hrl.common.modules.reward_modules import SparseRewardModule
from hrl.env.observation import EnvironmentObservation


class BufferModule:
    def __init__(self, eval_module: SparseRewardModule):
        self.actions: list[torch.Tensor] = []
        self.obs: list[EnvironmentObservation] = []
        self.goal: list[EnvironmentObservation] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.terminals: list[bool] = []
        self.eval_module: SparseRewardModule = eval_module

    def clear(self):
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.terminals.clear()
        self.obs.clear()
        self.goal.clear()

    def health(self):
        lengths = [
            len(self.actions),
            len(self.obs),
            len(self.goal),
            len(self.logprobs),
            len(self.rewards),
            len(self.values),
            len(self.terminals),
        ]
        return all(l == lengths[0] for l in lengths)

    def has_batch(self, batch_size: int):
        return len(self.actions) == batch_size

    def save(self, path: str, epoch: int):
        file_path = path + f"stats_epoch_{epoch}.pt"
        data = {}

        data["actions"] = torch.stack(self.actions)
        data["logprobs"] = torch.tensor(self.logprobs)
        data["values"] = torch.tensor(self.values)
        data["rewards"] = torch.tensor(self.rewards)
        data["terminals"] = torch.tensor(self.terminals)

        torch.save(data, file_path)

    def stats(self) -> tuple[float, float, float]:
        # Calculate episode statistics
        episode_rewards = []
        episode_lengths_batch = []
        episode_success = []

        current_episode_reward = 0
        current_episode_length = 0
        for _, (reward, terminal) in enumerate(zip(self.rewards, self.terminals)):
            current_episode_reward += reward
            current_episode_length += 1
            current_episode_success = (
                1.0
                if current_episode_reward
                == self.eval_module.config.success_reward
                + self.eval_module.config.step_reward * (current_episode_length - 1)
                else 0.0
            )
            print(
                f"Length: {current_episode_length}, Reward: {current_episode_reward}, Success: {current_episode_success}"
            )

            if terminal:
                episode_rewards.append(current_episode_reward)
                episode_lengths_batch.append(current_episode_length)
                episode_success.append(current_episode_success)
                current_episode_reward = 0
                current_episode_length = 0

        return (
            sum(episode_rewards),
            (
                sum(episode_lengths_batch) / len(episode_lengths_batch)
                if episode_lengths_batch
                else 0
            ),
            sum(episode_success) / len(episode_success) if episode_success else 0,
        )
