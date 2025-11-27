from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from src.hardware import device
from src.agents.agent import Agent, AgentConfig
from src.modules.buffer import Buffer
from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.networks.actor_critic import ActorCriticBase
from src.skills.skill import Skill
from loguru import logger

# Batch Size	The number of samples used
# in each training batch.	Larger batch sizes typically lead
# to more stable training but may require
# more memory and computational resources.
# Clip Range	The threshold for clipping
# the ratio of policy probabilities.	Clipping the ratio helps to prevent overly large policy updates, ensuring
# stability during training.
# Entropy Coefficient	A coefficient balancing between
# exploration and exploitation.	Higher values encourage more
# exploration, while lower values prioritize exploitation.
# The parameter controlling the balance between
# bias and variance in estimating advantages.	Lower values increase bias
# but reduce variance, leading to more stable training
# but potentially less accurate value estimates.
# Learning Rate	The rate at which the model’s
# parameters are updated during training.	A higher learning rate can accelerate learning but may lead to instability
# or divergence if too large.
# Log Std Init	The initial value for the logarithm
# of the standard deviation of the policy.	A higher initial value can encourage more exploration
# in the early stages of training.
# Epochs Number	The number of times the entire dataset is used during training.	Increasing the number of epochs allows for more passes through
# the data, utilizing data more efficiently.
# Steps Number	The number of steps taken when accumulating the dataset.	More step numbers allow for more information to be gathered per iteration,
# potentially leading to more efficient and stable learning.
# Normalize Advantages	Whether to normalize advantages
# before using them in training.	Normalization can help to stabilize training by scaling advantages to have
# a consistent impact on policy updates.
# Target KL	The target value for the Kullback-Leibler (KL)
# divergence between old and new policies.	Adjusting the target KL helps to regulate the magnitude of policy updates,
# promoting smoother learning.
# Value Coefficient	The coefficient balancing between the
# value loss and policy loss in the total loss function.	A higher value coefficient places more emphasis on the value function,
# potentially leading to more stable training.


@dataclass
class PPOAgentConfig(AgentConfig):
    # Default values
    eval: bool = False
    early_stop_patience: int = 5
    min_sampling_epochs: int = 10
    max_batches: int = 50
    saving_freq: int = 5  # Saving frequence of trained model
    save_stats: bool = True

    mini_batch_size: int = 64  # 64 # How many steps to use in each mini-batch
    learning_epochs: int = 30  # How many passes over the collected batch per update
    lr_annealing: bool = False
    learning_rate: float = 0.0003  # Step size for actor optimizer
    gamma: float = 0.99  # How much future rewards are worth today
    gae_lambda: float = 0.95  # Bias/variance trade‑off in advantage estimation
    eps_clip: float = 0.2  # How far the new policy is allowed to move from the old
    entropy_coef: float = 0.01  # Weight on the entropy bonus to encourage exploration
    value_coef: float = 0.5  # Weight on the critic (value) loss vs. the policy loss
    max_grad_norm: float = 0.5  # Threshold for clipping gradient norms
    target_kl: float | None = None  # (Optional) early stopping if KL


class PPOAgent(Agent):
    def __init__(
        self,
        config: PPOAgentConfig,
        policy_new: ActorCriticBase,
        policy_old: ActorCriticBase,
        buffer: Buffer,
        storage: Storage,
    ):
        ### Initialize hyperparameters
        self.config = config
        ### Initialize the agent
        self.buffer: Buffer = buffer
        self.storage: Storage = storage
        self.mse_loss = nn.MSELoss()
        self.policy_new: ActorCriticBase = policy_new.to(device)
        self.policy_old: ActorCriticBase = policy_old.to(device)
        self.optimizer = torch.optim.AdamW(
            self.policy_new.parameters(),
            lr=self.config.learning_rate,
        )
        if self.config.eval:
            self.policy_new.eval()
            self.policy_old.eval()

        ### Internal flags and counter
        self.current_epoch: int = 0
        self.best_success = 0
        self.epochs_since_improvement = 0

    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> Skill:
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(obs, goal)
        self.buffer.act_values(obs, goal, action, action_logprob, state_val)
        return self.storage.skills[int(action.item())]  # Can safely be accessed

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        return self.buffer.feedback(reward, success, terminal)

    def compute_gae(
        self,
        rewards: list[float],
        values: list[torch.Tensor],
        is_terminals: list[bool],
    ):
        advantages = []
        gae = 0
        values = values + [torch.tensor(0.0, device=device)]  # add dummy for V(s_{T+1})
        for step in reversed(range(len(rewards))):
            terminal = float(is_terminals[step])
            delta = (
                rewards[step]
                + self.config.gamma * values[step + 1] * (1 - terminal)
                - values[step]
            )
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * (1 - terminal) * gae
            )
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )

    def learn(self) -> bool:
        # Saves batch values
        success_rate = self.buffer.save(
            self.storage.buffer_saving_path,
            self.current_epoch,
        )

        ### Check for early stop (Plateau reached)
        if success_rate > self.best_success:  # New best success rate
            self.best_success = success_rate
            self.epochs_since_improvement = 0
            # Aditional save the new high.before new learning.
            self.save("best")
        else:
            self.epochs_since_improvement += 1

        # Check for improvement
        if (
            self.epochs_since_improvement >= self.config.early_stop_patience
            and self.config.min_sampling_epochs <= self.current_epoch
        ):
            logger.info(
                f"Early stopping training after {self.current_epoch} epochs because of no improvement in the last {self.config.early_stop_patience} epochs."
            )
            return True

        ### Preprocess batch values
        advantages, rewards = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.terminals,
        )

        # Check shapes
        assert (
            advantages.shape[0] == self.buffer.config.batch_size
        ), "Advantage shape mismatch"
        assert (
            rewards.shape[0] == self.buffer.config.batch_size
        ), "Reward shape mismatch"

        advantages = advantages.to(device)
        rewards = rewards.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_obs = self.buffer.current
        old_goal = self.buffer.goal
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        )

        ### Learning Rate Annealing
        if self.config.lr_annealing:
            progress = self.current_epoch / self.config.max_batches
            new_lr = self.config.learning_rate * (1.0 - progress)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        ### Training loop for network
        kl_divergence_stop = False
        for epoch in range(self.config.learning_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.buffer.config.batch_size)

            for start in range(
                0,
                self.buffer.config.batch_size,
                self.config.mini_batch_size,
            ):
                end = start + self.config.mini_batch_size
                mb_idx = indices[start:end]
                mb_idx_list = mb_idx.tolist()  # turn Tensor → Python list of ints
                # Decided to save observations as objects instead of tensors
                # Makes it easier to convert it based on network later on
                mb_obs = [old_obs[i] for i in mb_idx_list]
                mb_goal = [old_goal[i] for i in mb_idx_list]

                # mb_obs = old_obs[mb_idx]
                # mb_goal = old_goal[mb_idx]
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_rewards = rewards[mb_idx]

                # Evaluate policy
                logprobs, state_values, dist_new = self.policy_new.evaluate(
                    mb_obs, mb_goal, mb_actions
                )

                _, _, dist_old = self.policy_old.evaluate(mb_obs, mb_goal, mb_actions)

                assert logprobs.shape == mb_logprobs.shape, "Logprobs shape mismatch"
                assert (
                    state_values.shape == mb_rewards.shape
                ), "Value prediction shape mismatch"

                state_values = torch.squeeze(state_values)

                # Ratios
                ratios = torch.exp(logprobs - mb_logprobs.detach().to(device))

                # Surrogate loss
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.config.eps_clip,
                        1 + self.config.eps_clip,
                    )
                    * mb_advantages
                )

                # Calculate loss
                loss: torch.Tensor = (
                    -torch.min(surr1, surr2)
                    + self.config.value_coef * self.mse_loss(state_values, mb_rewards)
                    - self.config.entropy_coef * dist_new.entropy().mean()
                )

                ### Update gradients on mini-batch
                self.optimizer.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.policy_new.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Optional KL early stopping
                if self.config.target_kl is not None:
                    with torch.no_grad():
                        kl = torch.distributions.kl.kl_divergence(
                            dist_old, dist_new
                        ).mean()
                        if kl > self.config.target_kl:
                            kl_divergence_stop = True
                            break  # break minibatch loop
            if kl_divergence_stop:
                break

        ### Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_new.state_dict())

        # Clear buffer
        self.buffer.clear()

        # Update Epoch
        self.current_epoch += 1

        ### Regular saving based on saving frequence
        if self.current_epoch % self.config.saving_freq == 0:
            self.save()

        ### Stop Training
        if self.current_epoch == self.config.max_batches:
            print("Stopping Training cause of max epoch reached.")
            return True

        ### Continue Training otherwise
        return False

    def save(self, tag: str = ""):
        """
        Save the model to the specified path.
        """
        if tag == "":
            checkpoint_path = (
                self.storage.agent_saving_path
                + "model_cp_epoch_{}.pth".format(
                    self.current_epoch,
                )
            )
        else:
            checkpoint_path = self.storage.agent_saving_path + "model_cp_{}.pth".format(
                tag,
            )

        logger.info(
            f"Saving weights to: {checkpoint_path} at epoch {self.current_epoch}"
        )
        # torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(
            {
                "model_state": self.policy_old.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
            },
            checkpoint_path,
        )

    def metadata(self) -> dict:
        return {
            "mini_batch_size": self.config.mini_batch_size,
            "learning_epochs": self.config.learning_epochs,
            "lr_annealing": self.config.lr_annealing,
            "learning_rate": self.config.learning_rate,
            "gamma": self.config.gamma,
            "gae_lambda": self.config.gae_lambda,
            "eps_clip": self.config.eps_clip,
            "entropy_coef": self.config.entropy_coef,
            "value_coef": self.config.value_coef,
            "max_grad_norm": self.config.max_grad_norm,
            **(
                {"target_kl": self.config.target_kl}
                if self.config.target_kl is not None
                else {}
            ),
        }

    def metrics(self) -> dict[str, float]:
        return self.buffer.metrics()

    def weights(self) -> dict[str, np.ndarray]:
        return {
            f"weights/{name.replace('.', '/')}": param.data.cpu().numpy()
            for name, param in self.policy_new.named_parameters()
        }
