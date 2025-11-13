from dataclasses import dataclass

from src.core.skills.skill import BaseSkill
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation


@dataclass
class SkillEvalConfig:
    p_empty: float
    p_rand: float


class SkillEvalExperiment:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: SkillEvalConfig, env: CalvinEnvironment):
        # We sort based on Id for the baseline network to be consistent
        self.config = config
        self.env = env

    def step(self, skill: BaseSkill) -> CalvinObservation:
        return self.env.step(skill)

    def reset(self, skill: BaseSkill) -> tuple[CalvinObservation, CalvinObservation]:
        return self.env.reset(skill)

    def evaluate(self, skill: BaseSkill) -> tuple[float, bool]:
        return self.env.evaluate(skill)

    def close(self):
        self.env.close()
