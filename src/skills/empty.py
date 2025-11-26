import numpy as np
import torch
from calvin_env_modified.envs.observation import CalvinEnvObservation
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


class EmptySkill(Skill):
    def __init__(self):
        super().__init__(name="EmptySkill", id=-1)

    def reset(self, goal: StateValueDict, env: object):
        pass

    def predict(
        self,
        current: CalvinEnvObservation,
    ) -> np.ndarray | None:
        return None
