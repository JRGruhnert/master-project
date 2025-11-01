import torch

from src.core.modules.reward_module import RewardModule
from src.core.observation import BaseObservation


class BufferModule:
    def __init__(self, eval_module: RewardModule, batch_size: int):
        self.current: list[BaseObservation] = []
        self.goal: list[BaseObservation] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.terminals: list[bool] = []
        self.eval_module: RewardModule = eval_module
        self.batch_size: int = batch_size

    def clear(self):
        self.current.clear()
        self.goal.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.terminals.clear()

    def health(self):
        lengths = [
            len(self.current),
            len(self.goal),
            len(self.actions),
            len(self.logprobs),
            len(self.rewards),
            len(self.values),
            len(self.terminals),
        ]
        return all(l == lengths[0] for l in lengths)

    def save(self, path: str, epoch: int):
        assert self.health(), "Buffer lengths are inconsistent!"

        file_path = path + f"stats_epoch_{epoch}.pt"
        data = {
            "actions": torch.stack(self.actions),
            "logprobs": torch.tensor(self.logprobs),
            "values": torch.tensor(self.values),
            "rewards": torch.tensor(self.rewards),
            "terminals": torch.tensor(self.terminals),
        }
        torch.save(data, file_path)

    def act_values(
        self,
        current: BaseObservation,
        goal: BaseObservation,
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
        current: BaseObservation,
        goal: BaseObservation,
        action: int,
    ):
        self.current.append(current)
        self.goal.append(goal)
        self.actions.append(torch.tensor(action))
        self.logprobs.append(torch.tensor(0.0))
        self.values.append(torch.tensor(0.0))

    def feedback(self, reward: float, terminal: bool) -> bool:
        self.rewards.append(reward)
        self.terminals.append(terminal)
        assert self.health(), "Buffer lengths are inconsistent!"
        # print(len(self.actions))
        # print(self.batch_size)
        return len(self.actions) == self.batch_size

    def stats(self) -> tuple[float, float, float]:
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
            current_episode_success = (
                1.0 if reward == self.eval_module.config.success_reward else 0.0
            )

            if i == len(self.rewards) - 1 or self.terminals[i]:
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
