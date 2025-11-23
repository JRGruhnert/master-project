import torch
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


class EmptySkill(Skill):
    def __init__(self):
        super().__init__(name="EmptySkill", id=-1)

    def reset(self):
        pass

    def predict(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> torch.Tensor | None:
        return None
