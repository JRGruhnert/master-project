from dataclasses import dataclass
import torch
from torch import nn
import torch
from hrl.common.buffer import BufferModule
from hrl.common.storage import StorageModule
from tapas_gmm.utils.select_gpu import device
from hrl.env.observation import EnvironmentObservation
from hrl.networks.actor_critic import ActorCriticBase
from hrl.common.skill import Skill


@dataclass
class HRLAgentConfig:
    # Default values
    early_stop_patience: int = 5
    min_sampling_epochs: int = 10
    max_batches: int = 50
    saving_freq: int = 5  # Saving frequence of trained model

    save_stats: bool = True
    batch_size: int = 2048
    mini_batch_size: int = 64  # 64 # How many steps to use in each mini-batch
    learning_epochs: int = 50  # How many passes over the collected batch per update
    lr_annealing: bool = False
    lr_actor: float = 0.0003  # Step size for actor optimizer
    lr_critic: float = 0.0003  # NOTE unused # Step size for critic optimizer
    gamma: float = 0.99  # How much future rewards are worth today
    gae_lambda: float = 0.95  # Bias/variance trade‑off in advantage estimation
    eps_clip: float = 0.2  # How far the new policy is allowed to move from the old
    entropy_coef: float = 0.01  # Weight on the entropy bonus to encourage exploration
    value_coef: float = 0.5  # Weight on the critic (value) loss vs. the policy loss
    max_grad_norm: float = 0.5  # Threshold for clipping gradient norms
    target_kl: float | None = (
        None  # (Optional) early stopping if KL divergence gets too large
    )
    eval: bool = False


class HRLAgent:

    def __init__(
        self,
        config: HRLAgentConfig,
        model: ActorCriticBase,
        buffer_module: BufferModule,
        storage_module: StorageModule,
    ):
        # Hyperparameters
        self.config = config
        ### Initialize the agent
        self.buffer_module: BufferModule = buffer_module
        self.mse_loss = nn.MSELoss()
        self.policy_new: ActorCriticBase = model.to(device)
        self.policy_old: ActorCriticBase = model.to(device)
        self.storage_module = storage_module
        self.optimizer = torch.optim.AdamW(
            self.policy_new.parameters(),
            lr=self.config.lr_actor,
        )

        ### Internal flags and counter
        self.current_epoch: int = 0
        self.best_success = 0
        self.epochs_since_improvement = 0

    def act(
        self,
        obs: EnvironmentObservation,
        goal: EnvironmentObservation,
    ) -> Skill:
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(
                obs, goal, self.config.eval
            )
        self.buffer_module.obs.append(obs)
        self.buffer_module.goal.append(goal)
        self.buffer_module.actions.append(action)
        self.buffer_module.logprobs.append(action_logprob)
        self.buffer_module.values.append(state_val)
        return self.storage_module.skills[action.item()]  # Can safely be accessed

    def feedback(self, reward: float, terminal: bool):
        self.buffer_module.rewards.append(reward)
        self.buffer_module.terminals.append(terminal)
        return self.buffer_module.has_batch(self.config.batch_size)

    def compute_gae(
        self,
        rewards: list[float],
        values: list[torch.Tensor],
        is_terminals: list[float],
    ):
        advantages = []
        gae = 0
        values = values + [0]  # add dummy for V(s_{T+1})
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.config.gamma * values[step + 1] * (1 - is_terminals[step])
                - values[step]
            )
            gae = (
                delta
                + self.config.gamma
                * self.config.gae_lambda
                * (1 - is_terminals[step])
                * gae
            )
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(
            returns, dtype=torch.float32
        )

    def save_buffer(self):
        self.buffer_module.save(
            self.storage_module.buffer_saving_path, self.current_epoch
        )

    def learn(self) -> bool:
        assert self.buffer_module.health(), "Rollout buffer not in sync"
        assert (
            len(self.buffer_module.obs) == self.config.batch_size
        ), "Batch size mismatch"

        # Saves batch values
        if self.config.save_stats:
            self.save_buffer()

        total_reward, episode_length, success_rate = self.buffer_module.stats()

        ### Check for early stop (Plateau reached)
        if success_rate > self.best_success + 1e-2:  # small threshold
            self.best_success = success_rate
            self.epochs_since_improvement = 0
            # Aditional save the new highscore before new learning.
            self.save("best")
        else:
            self.epochs_since_improvement += 1

        if (
            self.epochs_since_improvement >= self.config.early_stop_patience
            and self.config.min_sampling_epochs <= self.current_epoch
        ):
            print("Aborting Training cause of no improvement.")
            return True

        ### Preprocess batch values
        advantages, rewards = self.compute_gae(
            self.buffer_module.rewards,
            self.buffer_module.values,
            self.buffer_module.terminals,
        )

        # Check shapes
        assert advantages.shape[0] == self.config.batch_size, "Advantage shape mismatch"
        assert rewards.shape[0] == self.config.batch_size, "Reward shape mismatch"

        advantages = advantages.to(device)
        rewards = rewards.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_obs = self.buffer_module.obs
        old_goal = self.buffer_module.goal
        old_actions = (
            torch.squeeze(torch.stack(self.buffer_module.actions, dim=0))
            .detach()
            .to(device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer_module.logprobs, dim=0))
            .detach()
            .to(device)
        )

        ### Learning Rate Annealing
        if self.config.lr_annealing:
            progress = self.current_epoch / self.config.max_batches
            new_lr = self.config.lr_actor * (1.0 - progress)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        ### Training loop for network
        kl_divergence_stop = False
        for epoch in range(self.config.learning_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.config.batch_size)

            for start in range(0, self.config.batch_size, self.config.mini_batch_size):
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
                logprobs, state_values, dist_entropy = self.policy_new.evaluate(
                    mb_obs, mb_goal, mb_actions
                )

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
                    - self.config.entropy_coef * dist_entropy
                )

                ### Update gradients on mini-batch
                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(
                    self.policy_new.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                # Optional KL early stopping
                if self.config.target_kl is not None:
                    with torch.no_grad():
                        kl = (mb_logprobs - logprobs).mean()
                        if kl > self.config.target_kl:
                            kl_divergence_stop = True
                            break  # break minibatch loop
            if kl_divergence_stop:
                break

        ### Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy_new.state_dict())

        # Clear buffer
        self.buffer_module.clear()

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

    def save(self, tag: str = None):
        """
        Save the model to the specified path.
        """
        if tag:
            print(f"Saving tagged Checkpoint: {tag}")
        else:
            print("Saving Regular Checkpoint!")
        if tag is None:
            checkpoint_path = (
                self.storage_module.agent_saving_path
                + "model_cp_epoch_{}.pth".format(
                    self.current_epoch,
                )
            )
        else:
            checkpoint_path = (
                self.storage_module.agent_saving_path
                + "model_cp_{}.pth".format(
                    tag,
                )
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

    def load(self):
        """
        Load the model from the specified path.
        """
        if "baseline" in self.storage_module.checkpoint_path.lower():
            self.load_baseline()
        else:
            self.load_gnn()

    def load_gnn(self):
        """
        Load the model from the specified path.
        """
        print("🔄 Loading GNN checkpoint...")
        checkpoint = torch.load(self.storage_module.checkpoint_path, map_location="cpu")

        self.policy_old.load_state_dict(checkpoint["model_state"])
        self.policy_new.load_state_dict(checkpoint["model_state"])
        # self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        # if keep_epoch:
        #    self.current_epoch = checkpoint["epoch"]  # redundant, but explicit

    def load_baseline(self):
        """Load state_dict and expand dimensions where needed"""
        print("🔄 Loading baseline checkpoint...")
        # Load the checkpoint
        checkpoint = torch.load(self.storage_module.checkpoint_path, map_location="cpu")
        old_state_dict: dict[str, torch.Tensor] = (
            checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        )

        # Get current model state dict
        new_state_dict = self.policy_old.state_dict()

        # Copy compatible weights
        for name, old_param in old_state_dict.items():
            if name in new_state_dict:
                new_param = new_state_dict[name]

                # Check if dimensions match
                if old_param.shape == new_param.shape:
                    # Direct copy
                    new_state_dict[name] = old_param
                elif len(old_param.shape) == len(new_param.shape):
                    # Same number of dimensions, try partial copy
                    new_state_dict[name] = self.expand_tensor_dims(
                        old_param, new_param.shape
                    )
                else:
                    print(
                        f"Skipping {name}: incompatible shapes {old_param.shape} -> {new_param.shape}"
                    )

        # Load the modified state dict
        self.policy_old.load_state_dict(new_state_dict, strict=True)
        self.policy_new.load_state_dict(new_state_dict, strict=True)

    def expand_tensor_dims(self, old_tensor, target_shape):
        """Expand tensor dimensions by copying/padding"""
        old_shape = old_tensor.shape
        new_tensor = torch.zeros(target_shape, dtype=old_tensor.dtype)

        # Copy the overlapping dimensions
        slices = []
        for i, (old_dim, new_dim) in enumerate(zip(old_shape, target_shape)):
            if old_dim <= new_dim:
                slices.append(slice(0, old_dim))
            else:
                slices.append(slice(0, new_dim))

        new_tensor[tuple(slices)] = old_tensor[tuple(slices)]
        return new_tensor
