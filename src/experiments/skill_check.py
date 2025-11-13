from src.core.skills.skill import BaseSkill
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation


class SkillCheckExperiment:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, env: CalvinEnvironment):
        # We sort based on Id for the baseline network to be consistent
        self.env = env

    def step(self, skill: BaseSkill) -> CalvinObservation:
        return self.env.step(skill)

    def reset(self, skill: BaseSkill) -> tuple[CalvinObservation, CalvinObservation]:
        return self.env.reset(skill)

    def eval_start(self, skill: BaseSkill) -> bool:
        return self.env.evaluate_skill(skill.precons)

    def eval_end(self, skill: BaseSkill) -> bool:
        return self.env.evaluate_skill(skill.postcons)

    def close(self):
        self.env.close()
