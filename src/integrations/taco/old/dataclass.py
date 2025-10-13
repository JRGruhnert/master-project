from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import torch
from typing import Dict, List, Optional
from calvin_env_modified.envs.observation import (
    CalvinEnvObservation,
)
from src.integrations.taco.transform_manager import CalvinTransformManager
from tacorl.datamodule.dataset.play_dataset import PlayDataset
from tacorl.utils.transforms import TransformManager


class EnvironmentSamplingDataset(Dataset):
    """
    Dataset that generates samples on-demand from environment interaction
    Perfect for evaluation where tasks are sampled by env.reset()
    """

    def __init__(
        self,
        env,
        modalities: DictConfig,
        action_type: str = "rel_actions_world",
        max_episode_length: int = 300,
        transform_manager: TransformManager = None,
        transf_type: str = "train",
        include_goal: bool = True,
        policy=None,  # Policy to generate actions (can be random, expert, or trained)
        num_virtual_episodes: int = 1000,  # Virtual length for dataset
    ):
        self.env = env
        self.modalities = modalities
        self.action_type = action_type
        self.max_episode_length = max_episode_length
        self.transform_manager = transform_manager
        self.transf_type = transf_type
        self.include_goal = include_goal
        self.policy = policy
        self.num_virtual_episodes = num_virtual_episodes

    def __len__(self) -> int:
        # Return virtual length - each call generates new episode
        return self.num_virtual_episodes

    def __getitem__(self, idx: int) -> Dict:
        """Generate a fresh episode on every call"""
        return self._generate_episode_on_demand()

    def _generate_episode_on_demand(self) -> Dict:
        """
        Generate a complete episode by interacting with the environment
        This is called every time the dataset is accessed
        """
        # Reset environment - samples new task
        obs, info = self.env.reset()

        observations = []
        actions = []

        for step in range(self.max_episode_length):
            # Convert observation format
            if isinstance(obs, CalvinEnvObservation):
                obs_dict = self.convert_calvin_obs_to_dict(obs)
            else:
                obs_dict = obs

            observations.append(obs_dict)

            # Get action from policy (this is where your agent acts!)
            action = self._get_action(obs_dict, info, step)
            actions.append(action)

            # Step environment with agent's action
            obs, reward, done, truncated, info = self.env.step(action)

            # Break if episode ends
            if done or truncated:
                break

        # Format as expected by TACO-RL
        sequence = self._format_episode_sequence(observations, actions, info)

        # Apply transforms
        if self.transform_manager:
            sequence = self.transform_manager(sequence, transf_type=self.transf_type)

        return sequence

    def _get_action(self, obs_dict: Dict, info: Dict, step: int) -> np.ndarray:
        """
        Get action from your agent/policy
        This is where your trained model or policy acts!
        """
        if self.policy is not None:
            # Use your trained policy/agent
            return self.policy.get_action(obs_dict, info)
        else:
            # Fallback to random actions for testing
            return self.env.action_space.sample()

    def _format_episode_sequence(
        self, observations: List[Dict], actions: List[np.ndarray], info: Dict
    ) -> Dict:
        """Format episode to match PlayDataset structure"""

        # Stack observations by modality
        states = {}
        for modality in self.modalities:
            if modality != self.action_type and "action" not in modality:
                if observations and modality in observations[0]:
                    states[modality] = torch.stack(
                        [torch.tensor(obs[modality]) for obs in observations]
                    )

        # Stack actions
        actions_tensor = torch.stack([torch.tensor(action) for action in actions])

        sequence = {
            "states": states,
            "actions": actions_tensor,
            "idx": np.random.randint(0, 10000),  # Random idx since it's generated
            "window_size": len(observations),
        }

        # Add goal if needed
        if self.include_goal:
            sequence["goal"] = self._extract_goal_from_episode(observations, info)
            sequence["disp"] = 1  # Or however you want to handle displacement

        # Add state info for compatibility
        sequence["state_info"] = self._extract_state_info(observations)

        return sequence

    def _extract_goal_from_episode(self, observations: List[Dict], info: Dict) -> Dict:
        """Extract goal from the episode"""
        # Option 1: Goal from environment info
        if "goal" in info:
            goal_obs = info["goal"]
            if isinstance(goal_obs, CalvinEnvObservation):
                return self.convert_calvin_obs_to_dict(goal_obs)
            return goal_obs

        # Option 2: Final state as goal
        if observations:
            return {
                modality: observations[-1][modality]
                for modality in self.modalities
                if modality in observations[-1] and "action" not in modality
            }

        return {}

    def convert_calvin_obs_to_dict(self, obs: CalvinEnvObservation) -> Dict:
        """Convert CalvinEnvObservation to dict format"""
        obs_dict = {}

        # Extract available modalities
        for modality in self.modalities:
            if "action" not in modality and hasattr(obs, modality):
                obs_dict[modality] = getattr(obs, modality)

        return obs_dict

    def _extract_state_info(self, observations: List[Dict]) -> Dict:
        """Extract state info for compatibility"""
        return {
            "robot_obs": torch.stack(
                [
                    torch.tensor(obs.get("robot_obs", np.zeros(7)))
                    for obs in observations
                ]
            ),
            "scene_obs": torch.stack(
                [
                    torch.tensor(obs.get("scene_obs", np.zeros(24)))
                    for obs in observations
                ]
            ),
        }
