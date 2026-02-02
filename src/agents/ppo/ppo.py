from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import wandb
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

# from torch.utils.data import DataLoader, Dataset

# from tapas.build.lib.tapas_gmm import dataset


# class PPOBufferDataset(Dataset):
#    def __init__(self, obs, goal, actions, logprobs, advantages, rewards):
#        self.data = list(zip(obs, goal, actions, logprobs, advantages, rewards))
#
#    def __len__(self):
#        return len(self.data)
#    def __getitem__(self, idx):
#        return self.data[idx]


@dataclass
class PPOAgentConfig(AgentConfig):
    # Default values
    retrain: bool = False
    eval: bool = False
    early_stop_patience: int = 5
    use_ema_for_early_stopping: bool = True
    ema_smoothing_factor: float = 0.1
    min_batches: int = 20
    max_batches: int = 100
    saving_freq: int = 5  # Saving frequence of trained model
    save_stats: bool = True

    mini_batch_size: int = 64  # 64 # How many steps to use in each mini-batch
    learning_epochs: int = 5  # How many passes over the collected batch per update
    lr_annealing: bool = False
    learning_rate: float = 0.0003  # Step size for actor optimizer
    gamma: float = 0.99  # How much future rewards are worth today
    gae_lambda: float = 0.95  # Bias/variance trade‑off in advantage estimation
    eps_clip: float = 0.2  # How far the new policy is allowed to move from the old
    entropy_coef: float = 0.02  # Weight on the entropy bonus to encourage exploration
    critic_coef: float = 0.5  # Weight on the critic (value) loss vs. the policy loss
    max_grad_norm: float = 0.5  # Threshold for clipping gradient norms
    target_kl: float | None = 0.02  # (Optional) early stopping if KL
    clip_value_loss: bool = True


class ProgressWatcher:
    def __init__(
        self,
        patience: int,
        min_batches: int,
        max_batches: int,
        use_ema: bool,
        smoothing_factor: float = 0.1,
    ):
        self.patience = patience
        self.min_batches = min_batches
        self.max_batches = max_batches
        self.use_ema = use_ema
        self.smoothing_factor = smoothing_factor
        self.best_value = -float("inf")
        self.counter = 0
        self.ema = None

    def update(self, metric: float, current_batch: int) -> bool:
        """
        Determines if training should stop and whether the metric is a new high.
        Returns:
            should_stop (bool): Whether early stopping should trigger.
            is_new_high (bool): Whether the current metric is a new high.
        """
        if self.use_ema:
            should_stop = self._ema_check(metric)
        else:
            should_stop = self._metric_check(metric)

        # Enforce min and max batch constraints
        if current_batch < self.min_batches:
            return False
        elif current_batch >= self.max_batches:
            return True
        else:
            return should_stop

    def _ema_check(self, metric: float) -> bool:
        # Update EMA
        if self.ema is None:
            self.ema = metric  # Initialize EMA with the first metric value
        else:
            self.ema = (
                self.smoothing_factor * metric + (1 - self.smoothing_factor) * self.ema
            )

        return self._metric_check(self.ema)

    def _metric_check(self, metric: float) -> bool:
        if metric > self.best_value:
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


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

        self.stop_watcher = ProgressWatcher(
            patience=self.config.early_stop_patience,
            min_batches=self.config.min_batches,
            max_batches=self.config.max_batches,
            use_ema=self.config.use_ema_for_early_stopping,
            smoothing_factor=self.config.ema_smoothing_factor,
        )
        if self.config.eval:
            self.policy_new.eval()
            self.policy_old.eval()

        ### Internal flags and dynamic values
        self._current_epoch: int = 0
        self._best_success: float = 0.0
        self._metrics: dict = {}

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

        ### Saves batch values
        batch_success_rate = self.buffer.save(
            self.storage.buffer_saving_path,
            self._current_epoch,
        )

        # Update best success rate
        if self._best_success <= batch_success_rate:
            self._best_success = batch_success_rate
            logger.info(
                f"Success rate {batch_success_rate:.4f} at epoch {self._current_epoch}. Saving best model."
            )
            self.save("best")

        ### Additional logging before clearing buffer and updating network
        self._metrics.update({"stats/best_success_rate": self._best_success})
        self._metrics.update(self.buffer.metrics())
        self._metrics.update(
            {
                f"weights/{name.replace('.', '/')}": wandb.Histogram(
                    param.data.cpu().numpy()
                )
                for name, param in self.policy_new.named_parameters()
            }
        )

        # Check batch success rate for early stopping
        if self.stop_watcher.update(batch_success_rate, self._current_epoch):
            logger.info(
                f"Early stopping training after {self._current_epoch} epochs because of no improvement in the smoothed success rate."
            )
            return True

        ### Preprocess batch values
        advantages, returns = self.compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.terminals,
        )

        # Check shapes
        assert (
            advantages.shape[0] == self.buffer.config.steps
        ), "Advantages shape mismatch"
        assert returns.shape[0] == self.buffer.config.steps, "Returns shape mismatch"

        advantages = advantages.to(device)
        returns = returns.to(device)
        old_obs = self.buffer.current
        old_goal = self.buffer.goal
        old_actions = (
            torch.stack(self.buffer.actions, dim=0).detach().to(device).squeeze(-1)
        )

        old_logprobs = (
            torch.stack(self.buffer.logprobs, dim=0).detach().to(device).squeeze(-1)
        )

        ### Create DataLoader
        # dataset = PPOBufferDataset(
        #    old_obs, old_goal, old_actions, old_logprobs, advantages, returns
        # )
        # dataloader = DataLoader(
        #    dataset,
        #    batch_size=self.config.mini_batch_size,
        #    shuffle=True,
        # )

        ### Learning Rate Annealing
        if self.config.lr_annealing:
            progress = self._current_epoch / self.config.max_batches
            new_lr = self.config.learning_rate * (1.0 - progress)
            self._metrics.update({"ppo/learning_rate": new_lr})
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        ### Training loop for network
        kl_divergence_stop = False
        for epoch in range(self.config.learning_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.buffer.config.steps)

            for start in range(
                0,
                self.buffer.config.steps,
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
                mb_returns = returns[mb_idx]

                # Normalize advantages per minibatch
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Evaluate policy
                logprobs, state_values, dist_new = self.policy_new.evaluate(
                    mb_obs, mb_goal, mb_actions
                )

                assert logprobs.shape == mb_logprobs.shape, "Logprobs shape mismatch"
                assert (
                    state_values.shape == mb_returns.shape
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

                # Value loss (with optional clipping)
                if self.config.clip_value_loss:
                    mb_old_values = (
                        torch.stack([self.buffer.values[i] for i in mb_idx_list])
                        .squeeze()
                        .to(device)
                    )
                    values_pred = mb_old_values + torch.clamp(
                        state_values - mb_old_values,
                        -self.config.eps_clip,
                        self.config.eps_clip,
                    )
                    value_loss = self.mse_loss(values_pred, mb_returns)
                else:
                    value_loss = self.mse_loss(state_values, mb_returns)

                # Calculate loss
                loss: torch.Tensor = (
                    -torch.min(surr1, surr2)
                    + self.config.critic_coef * value_loss
                    - self.config.entropy_coef * dist_new.entropy().mean()
                )

                ### Update gradients on mini-batch
                self.optimizer.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.policy_new.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                # Collect metrics (will keep last minibatch values)
                with torch.no_grad():
                    log_ratio = logprobs - mb_logprobs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    clip_fraction = (
                        ((ratios - 1).abs() > self.config.eps_clip).float().mean()
                    )
                    entropy = dist_new.entropy().mean()
                    policy_loss = (-torch.min(surr1, surr2)).mean()

                    self._metrics.update(
                        {
                            "ppo/policy_loss": policy_loss.item(),
                            "ppo/value_loss": value_loss.item(),
                            "ppo/entropy": entropy.item(),
                            "ppo/approx_kl": approx_kl.item(),
                            "ppo/clip_fraction": clip_fraction.item(),
                            "ppo/total_loss": loss.mean().item(),
                        }
                    )
                    # Optional KL early stopping
                    if self.config.target_kl is not None:
                        if approx_kl > self.config.target_kl:
                            kl_divergence_stop = True
                            break  # break minibatch loop
            if kl_divergence_stop:
                break

        # Computes explained variance over full batch
        with torch.no_grad():
            all_values = torch.stack(self.buffer.values).squeeze().to(device)
            var_returns = returns.var()
            if var_returns > 0:
                explained_var = (1 - (returns - all_values).var() / var_returns).item()
            else:
                explained_var = 0.0
            self._metrics.update({"ppo/explained_variance": explained_var})

        ### Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_new.state_dict())

        # Clear buffer
        self.buffer.clear()

        # Update Epoch
        self._current_epoch += 1

        ### Regular saving based on saving frequence
        if self._current_epoch % self.config.saving_freq == 0:
            self.save()

        ### Stop Training
        if self._current_epoch == self.config.max_batches:
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
                    self._current_epoch,
                )
            )
        else:
            checkpoint_path = self.storage.agent_saving_path + "model_cp_{}.pth".format(
                tag,
            )

        logger.info(
            f"Saving weights to: {checkpoint_path} at epoch {self._current_epoch}"
        )
        # torch.save(self.policy_old.state_dict(), checkpoint_path)
        torch.save(
            {
                "model_state": self.policy_old.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self._current_epoch,
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
            "critic_coef": self.config.critic_coef,
            "max_grad_norm": self.config.max_grad_norm,
            "clip_value_loss": self.config.clip_value_loss,
            **(
                {"target_kl": self.config.target_kl}
                if self.config.target_kl is not None
                else {}
            ),
        }

    def metrics(self) -> dict:
        return self._metrics
