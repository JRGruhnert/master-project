from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import re
import torch

from tapas_gmm.env.calvin import Calvin, CalvinConfig

from hrl.state.state import State
from hrl.skill.skill import Skill
from hrl.observation.observation import MPObservation


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


@dataclass
class MasterEnvConfig:
    sim_env: CalvinConfig
    debug_vis: bool
    # Reward Settings
    reward_mode: RewardMode = RewardMode.SPARSE
    max_reward: float = 100.0
    min_reward: float = -1.0


class MasterEnv(ABC):
    def __init__(
        self,
        config: MasterEnvConfig,
        states: list[State],
        tasks: list[Skill],
        max_steps: int,
    ):
        self.config = config
        self.states = states
        self.tasks = tasks
        self.sim_env = Calvin(config=config.sim_env)

        self.last_gripper_action = [1.0]  # open
        self.max_steps, self.steps_left = max_steps, max_steps  # Cached property
        self.terminal = False
        self.skill: Skill = None

    @abstractmethod
    def reset(self, skill: Skill = None) -> tuple[MPObservation, MPObservation]:
        """Resets the environment for a new episode. Returns the initial observation and goal observation."""
        raise NotImplementedError("Reset method not implemented yet.")

    @abstractmethod
    def give_control(self, skill: Skill, verbose: bool = False):
        viz_dict = {}  # TODO: Make available
        skill.policy.reset_episode(self.sim_env)
        # Batch prediction for the given observation
        try:
            prediction, _ = skill.policy.predict(
                self.make_tapas_format(self.current_calvin, skill, self.goal)
            )
            for action in prediction:
                if len(action.gripper) != 0:
                    self.last_gripper_action = action.gripper
                ee_action = np.concatenate(
                    (
                        action.ee,
                        self.last_gripper_action,
                    )
                )
                self.current_calvin, _, _, _ = self.sim_env.step(
                    ee_action, self.config.debug_vis, viz_dict
                )
                self.current = MPObservation(self.current_calvin)

        except FloatingPointError:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            print(f"Error happened!")

    @abstractmethod
    def close(self):
        raise NotImplementedError("Close method not implemented yet.")

    @abstractmethod
    def evaluate(self) -> tuple[float, bool]:
        if self.terminal:
            raise UserWarning(
                "Episode already ended. Please reset the evaluator with the new goal and state."
            )
        if self.config.reward_mode is RewardMode.SPARSE:
            if self.skill:
                if self.endposition_check(self.skill):
                    return self.config.max_reward, True
                else:
                    return self.config.min_reward, False
            if self.completion_check():
                return self.config.max_reward, True
            else:
                return self.config.min_reward, False if self.steps_left > 0 else True
        if self.config.reward_mode is RewardMode.ONOFF:
            raise NotImplementedError("Reward Mode not implemented.")
        if self.config.reward_mode is RewardMode.RANGE:
            raise NotImplementedError("Reward Mode not implemented.")

    def startposition_check(self, skill: Skill) -> bool:
        ##### Checking if start position is reached
        for state in self.states:
            if state.name in skill.task_parameters_keys:
                if skill.reversed:
                    value = skill.anti_task_parameters[state.name]
                else:
                    value = skill.task_parameters[state.name]
                start_reached = state.evaluate(
                    self.current.states[state.name],
                    value,
                )
            if not start_reached:
                return False  # Early exit if start position is not reached
        return True

    def endposition_check(self, skill: Skill) -> bool:
        ##### Checking if end position is reached
        for state in self.states:
            if state.name in skill.task_parameters_keys:
                if skill.reversed:
                    value = skill.anti_task_parameters[state.name]
                else:
                    value = skill.anti_task_parameters[state.name]
                end_reached = state.evaluate(
                    self.current.states[state.name],
                    value,
                )
            if not end_reached:
                return False  # Early exit if end position is not reached

    def completion_check(self) -> bool:
        ##### Checking if goal is reached
        for state in self.states:
            goal_reached = state.evaluate(
                self.current.states[state.name],
                self.goal.states[state.name],
            )
            # print(f"State {state.name} is {goal_reached}")
            if not goal_reached:
                return False  # Early exit if goal is already not reached
        return True

    @abstractmethod
    def make_tapas_format(self, obs: CalvinObservation, task: Skill = None, goal: MPObservation = None) -> SceneObservation:  # type: ignore
        """
        Convert the observation from the environment to a SceneObservation. This format is used for TAPAS.

        Returns
        -------
        SceneObservation
            The observation in common format as SceneObservation.
        """
        if obs.action is None:
            action = None
        else:
            action = torch.Tensor(obs.action)

        if obs.reward is None:
            reward = torch.Tensor([0.0])
        else:
            reward = torch.Tensor([obs.reward])
        joint_pos = torch.Tensor(obs.joint_pos)
        joint_vel = torch.Tensor(obs.joint_vel)
        ee_pose = torch.Tensor(obs.ee_pose)
        ee_state = torch.Tensor([obs.ee_state])
        camera_obs = {}
        for cam in obs.camera_names:
            rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255
            mask = obs.mask[cam].astype(int)

            camera_obs[cam] = SingleCamObservation(
                **{
                    "rgb": torch.Tensor(rgb),
                    "depth": torch.Tensor(obs.depth[cam]),
                    "mask": torch.Tensor(mask).to(torch.uint8),
                    "extr": torch.Tensor(obs.extr[cam]),
                    "intr": torch.Tensor(obs.intr[cam]),
                },
                batch_size=empty_batchsize,
            )

        multicam_obs = dict_to_tensordict(
            {"_order": CameraOrder._create(obs.camera_names)} | camera_obs
        )
        object_poses_dict = obs.object_poses
        object_states_dict = obs.object_states
        if task is not None and task.reversed:
            assert goal is not None, "Goal must be provided for reversed tasks."
            # NOTE: This is only a hack to make reversed tapas models work
            # TODO: Update this when possible
            # logger.debug(f"Overriding Tapas Task {task.name}")
            for state_name, state_value in task.overrides.items():
                match_position = re.search(r"(.+?)_(?:position)", state_name)
                match_rotation = re.search(r"(.+?)_(?:rotation)", state_name)
                match_scalar = re.search(r"(.+?)_(?:scalar)", state_name)
                if state_name == "ee_position":
                    ee_pose = torch.cat(
                        [
                            torch.Tensor(state_value),
                            ee_pose[3:],
                        ]
                    )
                elif state_name == "ee_rotation":
                    ee_pose = torch.cat(
                        [
                            ee_pose[:3],
                            torch.Tensor(state_value),
                        ]
                    )
                elif state_name == "ee_scalar":
                    ee_state = torch.Tensor(state_value)

                # TODO: Evaluate if goal state is correct here
                elif match_position:
                    temp_pos = self.states[0].area_tapas_override(
                        goal.states[f"{match_position.group(1)}_position"],
                        self.spawn_surfaces,
                    )
                    object_poses_dict[match_position.group(1)] = np.concatenate(
                        [
                            temp_pos.numpy(),
                            object_poses_dict[match_position.group(1)][3:],
                        ]
                    )
                elif match_rotation:
                    object_poses_dict[match_rotation.group(1)] = np.concatenate(
                        [
                            object_poses_dict[match_rotation.group(1)][:3],
                            goal.states[f"{match_rotation.group(1)}_rotation"].numpy(),
                        ]
                    )
                elif match_scalar:
                    object_states_dict[match_scalar.group(1)] = goal.states[
                        f"{match_scalar.group(1)}_scalar"
                    ].numpy()
                else:
                    raise ValueError(f"Unknown state name: {state_name}")

        object_poses = dict_to_tensordict(
            {
                name: torch.Tensor(pose)
                for name, pose in sorted(object_poses_dict.items())
            },
        )
        object_states = dict_to_tensordict(
            {
                name: torch.Tensor([state])
                for name, state in sorted(object_states_dict.items())
            },
        )

        return SceneObservation(
            feedback=reward,
            action=action,
            cameras=multicam_obs,
            ee_pose=ee_pose,
            gripper_state=ee_state,
            object_poses=object_poses,
            object_states=object_states,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            batch_size=empty_batchsize,
        )
