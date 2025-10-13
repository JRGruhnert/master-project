from calvin_env.envs.calvin_env import CalvinEnvObservation


import numpy as np


def calvin_old_to_new(obs: dict) -> CalvinEnvObservation:
    """
    Convert a dictionary compatible with Taco to a CalvinEnvObservation.
    """
    ee_pose = np.concatenate([obs["ee_position"], obs["ee_rotation"]])
    ee_state = obs["ee_scalar"][0]

    object_poses = {
        k.split("_")[0]: np.concatenate([v[:3], v[3:]])
        for k, v in obs.items()
        if k.endswith("_position") or k.endswith("_rotation")
    }
    object_states = {
        k.split("_")[0]: v[0] for k, v in obs.items() if k.endswith("_scalar")
    }

    return CalvinEnvObservation(
        ee_pose=ee_pose,
        ee_state=ee_state,
        object_poses=object_poses,
        object_states=object_states,
    )
