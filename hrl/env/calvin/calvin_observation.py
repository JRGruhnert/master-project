from functools import cached_property
import numpy as np
import torch

from hrl.env.observation import BaseObservation

from calvin_env.envs.calvin_env import CalvinEnvObservation


class CalvinObservation(BaseObservation):
    def __init__(self, obs: CalvinEnvObservation):
        # Build the tensor dict
        state_dict = {}
        state_dict["ee_position"] = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        state_dict["ee_rotation"] = torch.tensor(obs.ee_pose[-4:], dtype=torch.float32)
        state_dict["ee_scalar"] = torch.tensor(
            np.array([obs.ee_state]), dtype=torch.float32
        )

        for k, pose in obs.object_poses.items():
            state_dict[f"{k}_position"] = torch.tensor(pose[:3], dtype=torch.float32)
            state_dict[f"{k}_rotation"] = torch.tensor(pose[-4:], dtype=torch.float32)

        for k, val in obs.object_states.items():
            state_dict[f"{k}_scalar"] = torch.tensor(
                np.array([val]), dtype=torch.float32
            )

        # Initialize parent TensorDict with batch_size=() for single observations
        super().__init__(state_dict, batch_size=torch.Size([]))
