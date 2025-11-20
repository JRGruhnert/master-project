import torch
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


class AdjustSkill(Skill):
    def __init__(self):
        super().__init__(name="AdjustSkill", id=-1)

    def reset(self):
        pass

    def predict(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> torch.Tensor | None:
        return None

    def _to_skill_format(
        self,
        current: StateValueDict,
        goal: StateValueDict | None = None,
    ) -> dict:
        return {
            "skill_name": self.name,
            "skill_id": self.id,
            "parameters": {},
        }
