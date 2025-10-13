from dataclasses import dataclass
import random

from loguru import logger
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation


@dataclass
class PePrConfig:
    p_empty: float
    p_rand: float


class PePrExperiment:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: PePrConfig, env: CalvinEnvironment):
        # We sort based on Id for the baseline network to be consistent

        self.p_empty = config.p_empty
        self.p_rand = config.p_rand
        self.env = env
        num_skills = len(env.storage_module.skills)
        self.max_episode_length = int(
            num_skills + num_skills * self.p_empty + num_skills * self.p_rand
        )
        self.current_step = 0

    def step(self, skill) -> tuple[CalvinObservation, float, bool]:
        sample = random.random()
        if sample < self.p_empty:  # 0-p_empty>
            logger.warning("Taking Empty Step")
            pass
        elif sample < self.p_empty + self.p_rand:  # 0-p_empty + p_rand>
            logger.warning("Taking Random Step")
            self.env.step(random.choice(self.env.storage_module.skills))
        else:  # The rest
            self.env.step(skill)

    def reset(self) -> tuple[CalvinObservation, CalvinObservation]:
        self.current_step = 0
        return self.env.reset()

    def evaluate(self) -> tuple[float, bool]:
        reward, done = self.env.evaluate()
        terminal = True if self.current_step >= self.max_episode_length else done
        self.current_step += 1
        return reward, terminal

    def close(self):
        self.env.close()
