import numpy as np
import torch
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


class AdjustSkill(Skill):
    def __init__(self, step_size: float = 0.05):
        super().__init__(name="AdjustSkill", id=-1)
        self.step_size = step_size

    def reset(self):
        pass

    def predict(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> np.ndarray | None:
        # Get current and goal positions/rotations as numpy arrays
        curr_pos = current["ee_position"].cpu().numpy()
        goal_pos = goal["ee_position"].cpu().numpy()
        curr_rot = current["ee_rotation"].cpu().numpy()
        goal_rot = goal["ee_rotation"].cpu().numpy()

        # Compute direction vectors
        delta_pos = goal_pos - curr_pos
        delta_rot = goal_rot - curr_rot

        # Normalize and scale step
        if np.linalg.norm(delta_pos) > 1e-6:
            step_pos = curr_pos + self.step_size * delta_pos / np.linalg.norm(delta_pos)
        else:
            step_pos = curr_pos

        if np.linalg.norm(delta_rot) > 1e-6:
            step_rot = curr_rot + self.step_size * delta_rot / np.linalg.norm(delta_rot)
        else:
            step_rot = curr_rot

        # Concatenate position and rotation for action
        action = np.concatenate([step_pos, step_rot])
        return action
