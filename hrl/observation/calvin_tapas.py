import re
import numpy as np
import torch
from calvin_env.envs.observation import CalvinObservation

from hrl.observation.observation import EnvironmentObservation


class MPObservation(EnvironmentObservation):
    def __init__(
        self,
        obs: CalvinObservation,
    ):

        self._states: dict[str, torch.Tensor] = {}
        self._states["ee_position"] = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        self._states["ee_rotation"] = torch.tensor(
            obs.ee_pose[-4:], dtype=torch.float32
        )
        self._states["ee_scalar"] = torch.tensor([obs.ee_state], dtype=torch.float32)
        for k, pose in obs.object_poses.items():
            self._states[f"{k}_position"] = torch.tensor(pose[:3], dtype=torch.float32)
            self._states[f"{k}_rotation"] = torch.tensor(pose[-4:], dtype=torch.float32)

        for k, val in obs.object_states.items():
            self._states[f"{k}_scalar"] = torch.tensor([val], dtype=torch.float32)

    def to_skill_format(self, obs: CalvinObservation, task: Skill = None, goal: MPObservation = None) -> SceneObservation:  # type: ignore
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
