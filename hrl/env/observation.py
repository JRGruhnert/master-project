from functools import cached_property
from tensordict import TensorDict
import torch
from calvin_env.envs.observation import CalvinObservation


class EnvironmentObservation(TensorDict):
    def __init__(self, obs: CalvinObservation):
        # Build the tensor dict
        state_dict = {}
        state_dict["ee_position"] = torch.tensor(obs.ee_pose[:3], dtype=torch.float32)
        state_dict["ee_rotation"] = torch.tensor(obs.ee_pose[-4:], dtype=torch.float32)
        state_dict["ee_scalar"] = torch.tensor([obs.ee_state], dtype=torch.float32)

        for k, pose in obs.object_poses.items():
            state_dict[f"{k}_position"] = torch.tensor(pose[:3], dtype=torch.float32)
            state_dict[f"{k}_rotation"] = torch.tensor(pose[-4:], dtype=torch.float32)

        for k, val in obs.object_states.items():
            state_dict[f"{k}_scalar"] = torch.tensor([val], dtype=torch.float32)

        # Initialize parent TensorDict with batch_size=() for single observations
        super().__init__(state_dict, batch_size=torch.Size([]))

    @cached_property
    def states(self) -> dict[str, torch.Tensor]:
        """Backward compatibility - returns dict view."""
        return dict(self)
