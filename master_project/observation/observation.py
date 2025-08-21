import torch
import numpy as np
from typing import Dict, Union, Any, Optional
from calvin_env.envs.observation import CalvinObservation
from tapas_gmm.utils.observation import (
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)


class SceneObservationDict:
    """
    Dictionary-based wrapper around SceneObservation for efficient access.
    Provides both structured access and flattened dictionary interface.
    """
    
    def __init__(self, scene_obs: SceneObservation):
        self.scene_obs = scene_obs
        self._flat_cache: Optional[Dict[str, torch.Tensor]] = None
    
    def get_flat_dict(self) -> Dict[str, torch.Tensor]:
        """Get flattened dictionary representation of the scene observation."""
        if self._flat_cache is None:
            self._flat_cache = self._create_flat_dict()
        return self._flat_cache
    
    def _create_flat_dict(self) -> Dict[str, torch.Tensor]:
        """Create flattened dictionary from SceneObservation."""
        flat_dict = {}
        
        # Robot states
        if self.scene_obs.ee_pose is not None:
            flat_dict["ee_position"] = self.scene_obs.ee_pose[:3]
            flat_dict["ee_rotation"] = self.scene_obs.ee_pose[3:]
        
        if self.scene_obs.gripper_state is not None:
            flat_dict["ee_scalar"] = self.scene_obs.gripper_state
            
        if self.scene_obs.joint_pos is not None:
            flat_dict["joint_positions"] = self.scene_obs.joint_pos
            
        if self.scene_obs.joint_vel is not None:
            flat_dict["joint_velocities"] = self.scene_obs.joint_vel
        
        # Object poses
        if self.scene_obs.object_poses is not None:
            for obj_name, pose in self.scene_obs.object_poses.items():
                flat_dict[f"{obj_name}_position"] = pose[:3]
                flat_dict[f"{obj_name}_rotation"] = pose[3:]
        
        # Object states
        if self.scene_obs.object_states is not None:
            for obj_name, state in self.scene_obs.object_states.items():
                flat_dict[f"{obj_name}_scalar"] = state
        
        # Action and feedback
        if self.scene_obs.action is not None:
            flat_dict["action"] = self.scene_obs.action
            
        if self.scene_obs.feedback is not None:
            flat_dict["reward"] = self.scene_obs.feedback
            
        return flat_dict
    
    def __getitem__(self, key: str) -> torch.Tensor:
        """Dictionary-like access to flattened observation."""
        return self.get_flat_dict()[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in flattened observation."""
        return key in self.get_flat_dict()
    
    def keys(self):
        """Get all keys from flattened observation."""
        return self.get_flat_dict().keys()
    
    def items(self):
        """Get all items from flattened observation."""
        return self.get_flat_dict().items()


def create_scene_observation_from_calvin(obs: CalvinObservation) -> SceneObservation:
    """
    Create RL-Bench compatible SceneObservation from Calvin observation.
    This is the preferred way to convert Calvin observations.
    """
    # Handle action
    action = torch.tensor(obs.action) if obs.action is not None else None
    
    # Handle reward
    reward = torch.tensor([obs.reward]) if obs.reward is not None else torch.tensor([0.0])
    
    # Robot states
    joint_pos = torch.tensor(obs.joint_pos, dtype=torch.float32)
    joint_vel = torch.tensor(obs.joint_vel, dtype=torch.float32)
    ee_pose = torch.tensor(obs.ee_pose, dtype=torch.float32)
    ee_state = torch.tensor([obs.ee_state], dtype=torch.float32)
    
    # Camera observations
    camera_obs = {}
    for cam in obs.camera_names:
        rgb = obs.rgb[cam].transpose((2, 0, 1)) / 255.0  # Normalize to [0,1]
        mask = obs.mask[cam].astype(int) if obs.mask is not None else None
        
        cam_data = {"rgb": torch.tensor(rgb, dtype=torch.float32)}
        if mask is not None:
            cam_data["mask"] = torch.tensor(mask, dtype=torch.int32)
            
        camera_obs[cam] = SingleCamObservation(**cam_data)
    
    # Object poses
    object_poses = dict_to_tensordict({
        name: torch.tensor(pose, dtype=torch.float32)
        for name, pose in obs.object_poses.items()
    })
    
    # Object states  
    object_states = dict_to_tensordict({
        name: torch.tensor([state], dtype=torch.float32)
        for name, state in obs.object_states.items()
    })
    
    return SceneObservation(
        feedback=reward,
        action=action,
        cameras=camera_obs,
        ee_pose=ee_pose,
        gripper_state=ee_state,
        object_poses=object_poses,
        object_states=object_states,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        batch_size=empty_batchsize,
    )


class ObservationUtils:
    """Utility functions for working with SceneObservation and observation dictionaries."""

    @staticmethod
    def scene_obs_to_numpy(scene_obs: SceneObservation) -> Dict[str, np.ndarray]:
        """Convert SceneObservation tensors to numpy arrays."""
        obs_dict = SceneObservationDict(scene_obs)
        return {k: v.detach().cpu().numpy() for k, v in obs_dict.get_flat_dict().items()}

    @staticmethod
    def dict_to_numpy(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """Convert tensor dictionary to numpy dictionary."""
        return {k: v.detach().cpu().numpy() for k, v in obs_dict.items()}

    @staticmethod
    def to_device(obs_dict: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
        """Move all tensors to specified device."""
        return {k: v.to(device) for k, v in obs_dict.items()}

    @staticmethod
    def stack_flat_observations(obs_list: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Stack multiple flat observation dictionaries into batched tensors."""
        if not obs_list:
            return {}

        keys = obs_list[0].keys()
        return {k: torch.stack([obs[k] for obs in obs_list]) for k in keys}
    
    @staticmethod
    def stack_scene_observations(scene_obs_list: list[SceneObservation]) -> Dict[str, torch.Tensor]:
        """Stack multiple SceneObservations into batched flat dictionary."""
        flat_obs_list = [SceneObservationDict(obs).get_flat_dict() for obs in scene_obs_list]
        return ObservationUtils.stack_flat_observations(flat_obs_list)

    @staticmethod
    def extract_state_subset(obs_dict: Dict[str, torch.Tensor], state_names: list[str]) -> Dict[str, torch.Tensor]:
        """Extract subset of states from observation dictionary."""
        return {name: obs_dict[name] for name in state_names if name in obs_dict}


# Legacy function for backward compatibility
def create_scene_observation(obs: CalvinObservation) -> Dict[str, torch.Tensor]:
    """
    Legacy function - creates flat dictionary. 
    Use SceneObservationDict for RL-Bench compatibility.
    """
    scene_obs = create_scene_observation_from_calvin(obs)
    return SceneObservationDict(scene_obs).get_flat_dict()
