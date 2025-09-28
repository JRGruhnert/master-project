from dataclasses import dataclass
import os
import torch
from torch import nn
import torch
from hrl.common.buffer import RolloutBuffer
from hrl.networks import NetworkType, import_network
from tapas_gmm.utils.select_gpu import device
from hrl.observation.observation import EnvironmentObservation
from hrl.networks.actor_critic import ActorCriticBase
from hrl.state.state import State
from hrl.skill.skill import Skill
from ptflops import get_model_complexity_info
from torchinfo import summary
from torch_geometric.data import Batch


@dataclass
class AgentConfig:
    # Default values
    early_stop_patience: int = 5
    min_sampling_epochs: int = 10
    max_batches: int = 50
    saving_freq: int = 5  # Saving frequence of trained model
    saving_path: str = "results/"

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


class MasterAgent:
    def __init__(
        self,
        config: AgentConfig,
        nt: NetworkType,
        tag: str,
        states: list[State],
        tasks: list[Skill],
    ):
        # Hyperparameters
        self.config = config
        self.nt: NetworkType = nt
        Net = import_network(nt)
        print("Using network:", nt)
        ### Initialize the agent
        self.states: list[State] = states
        self.tasks: list[Skill] = tasks
        self.mse_loss = nn.MSELoss()
        self.buffer = RolloutBuffer()
        self.policy_new: ActorCriticBase = Net(states, tasks).to(device)
        self.policy_old: ActorCriticBase = Net(states, tasks).to(device)
        self.optimizer = torch.optim.AdamW(
            self.policy_new.parameters(),
            lr=self.config.lr_actor,
        )

        # param_count = sum(p.numel() for p in self.policy_new.parameters())
        # print(f"Total {self.nt.value} parameters: {param_count}")
        # if self.nt is NetworkType.GNN_V4:
        #    dummy_batch = Batch.from_data_list([obs, dummy_graph2])
        #    summary(self.policy_old, input_data=(dummy_batch, goal))
        # else:
        #    dummy_obs = torch.zeros(batch_size, obs_dim)
        #    dummy_goal = torch.zeros(batch_size, goal_dim)
        #    summary(self.policy_old, input_data=(dummy_obs, dummy_goal))

        ### Internal flags and counter
        self.waiting_feedback: bool = False
        self.current_epoch: int = 0
        # For early stopping
        self.best_success = 0
        self.epochs_since_improvement = 0

        ### Directory path for the agent (specified by name)
        self.directory_path = config.saving_path + nt.value + "/" + tag + "/"
        if not os.path.exists(self.directory_path):
            os.makedirs(self.directory_path)

        self.log_path = self.directory_path + "logs/"
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def act(
        self,
        obs: EnvironmentObservation,
        goal: EnvironmentObservation,
        eval: bool = False,
    ) -> Skill:
        if self.waiting_feedback:
            raise UserWarning(
                "The agent hasn't recieved any feedback of previous action yet. "
                "Learning will not work without feedback. Please call feedback() fafter every act() call."
            )

        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(obs, goal, eval)

        self.buffer.obs.append(obs)
        self.buffer.goal.append(goal)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(state_val)
        self.waiting_feedback = True
        return self.tasks[action.item()]  # Can safely be accessed

    def feedback(self, reward: float, terminal: bool):
        if not self.waiting_feedback:
            raise UserWarning(
                "The agent is recieving feedback without any action taken. "
                "Learning will not work without actions. Is this correct?"
            )

        self.buffer.rewards.append(reward)
        self.buffer.terminals.append(terminal)
        self.waiting_feedback = False
        return self.buffer.has_batch(self.config.batch_size)

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

    def save(self):
        self.buffer.save(self.log_path, self.current_epoch)

    def learn(self, verbose: bool = False) -> bool:
        assert self.buffer.health(), "Rollout buffer not in sync"
        assert len(self.buffer.obs) == self.config.batch_size, "Batch size mismatch"

        # Saves batch values
        if self.config.save_stats:
            self.save()

        total_reward, episode_length, success_rate = self.buffer.stats()
        if verbose:

            print(
                f"Total Reward: {total_reward} \t Episode Length: {episode_length:.2f} \t Success Rate: {success_rate:.3f}"
            )

            print(
                f"Called learning on new batch (Batch {self.current_epoch}). Updating gradients of the agent!"
            )

        ### Check for early stop (Plateau reached)
        if success_rate > self.best_success + 1e-2:  # small threshold
            self.best_success = success_rate
            self.epochs_since_improvement = 0
            # Aditional save the new highscore before new learning.
            self.save("best", verbose)
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
            self.buffer.rewards, self.buffer.values, self.buffer.terminals
        )

        # Check shapes
        assert advantages.shape[0] == self.config.batch_size, "Advantage shape mismatch"
        assert rewards.shape[0] == self.config.batch_size, "Reward shape mismatch"

        advantages = advantages.to(device)
        rewards = rewards.to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_obs = self.buffer.obs
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
            new_lr = self.config.lr_actor * (1.0 - progress)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            if verbose:
                print(
                    f"[Epoch {self.current_epoch}] Base LR: {new_lr:.6f} (progress={progress:.2f})"
                )

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
                            print(f"Early stopping at epoch {epoch} due to KL={kl:.4f}")
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
            self.save(verbose=verbose)

        ### Stop Training
        if self.current_epoch == self.config.max_batches:
            print("Stopping Training cause of max epoch reached.")
            return True

        ### Continue Training otherwise
        return False

    def save(self, tag: str = None, verbose: bool = False):
        """
        Save the model to the specified path.
        """
        if verbose:
            print("Saving Checkpoint!")
        if tag is None:
            checkpoint_path = self.directory_path + "model_cp_epoch_{}.pth".format(
                self.current_epoch,
            )
        else:
            checkpoint_path = self.directory_path + "model_cp_{}.pth".format(
                tag,
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

    def load(self, checkpoint_path: str):
        """
        Load the model from the specified path.
        """
        if self.nt in {
            NetworkType.BASELINE_TEST,
            NetworkType.BASELINE_V1,
            NetworkType.BASELINE_V2,
        }:
            self.load_baseline(checkpoint_path)
        else:
            self.load_gnn(checkpoint_path)

    def load_gnn(self, checkpoint_path: str):
        """
        Load the model from the specified path.
        """
        print(f"Loading checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self.policy_old.load_state_dict(checkpoint["model_state"])
        self.policy_new.load_state_dict(checkpoint["model_state"])
        # self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        # if keep_epoch:
        #    self.current_epoch = checkpoint["epoch"]  # redundant, but explicit

    def load_baseline(self, checkpoint_path: str):
        """Load state_dict and expand dimensions where needed"""

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
