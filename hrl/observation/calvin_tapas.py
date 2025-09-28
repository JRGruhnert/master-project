import torch
from calvin_env.envs.observation import CalvinObservation
from hrl.observation.observation import EnvironmentObservation


class MasterCalvinObs(EnvironmentObservation):
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
