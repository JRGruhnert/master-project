from dataclasses import dataclass
import random

from loguru import logger
from src.core.skills.skill import BaseSkill, EmptySkill
from src.experiments.experiment import Experiment
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation
from src.core.skills.tapas import TapasSkill


@dataclass
class PePrConfig:
    p_empty: float
    p_rand: float


class PePrExperiment(Experiment):
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: PePrConfig, env: CalvinEnvironment):
        # We sort based on Id for the baseline network to be consistent
        super().__init__(env)
        self.config = config
        num_skills = 6 if len(env.storage_module.skills) < 12 else 16
        self.max_episode_length = int(
            num_skills
            + num_skills * self.config.p_empty
            + num_skills * self.config.p_rand
        )
        # 6 or 16
        self.current_step = 0

    def step(self, skill: BaseSkill) -> CalvinObservation:
        sample = random.random()
        if sample < self.config.p_empty or isinstance(skill, EmptySkill):  # 0-p_empty>
            logger.warning("Taking Empty Step")
            pass
        elif sample < self.config.p_empty + self.config.p_rand:  # 0-p_empty + p_rand>
            logger.warning("Taking Random Step")
            self.current = self.env.step(random.choice(self.env.storage_module.skills))
        else:  # The rest
            self.current = self.env.step(skill)

        return self.current

    def sample(self) -> tuple[CalvinObservation, CalvinObservation]:
        self.current_step = 0
        self.current, goal = self.env.sample_task()
        return self.current, goal

    def evaluate(self) -> tuple[float, bool]:
        self.current_step += 1
        reward, done = self.env.evaluate()
        terminal = True if self.current_step >= self.max_episode_length else done
        return reward, terminal

    def close(self):
        self.env.close()
