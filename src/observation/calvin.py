import numpy as np
import torch

from src.observation.observation import StateValueDict

from calvin_env_modified.envs.observation import (
    CalvinEnvObservation,
)


class CalvinObservation(StateValueDict):
    """Just a simple wrapper around TensorDict for Calvin observations
    to be called everywhere consistently."""

    @classmethod
    def from_internal(cls, obs: CalvinEnvObservation) -> StateValueDict:
        """Create BaseObservation from regular dict"""
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
        return cls(state_dict)
