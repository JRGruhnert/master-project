from dataclasses import dataclass
import random

from loguru import logger
from hrl.env.calvin import CalvinEnvironment
from hrl.env.observation import EnvironmentObservation


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

    def step(self, skill) -> tuple[EnvironmentObservation, float, bool]:
        sample = random.random()
        print(f"Random Sample: {sample}")  # Debug output
        if sample < self.p_empty:  # 0-p_empty>
            logger.warning("Taking Empty Step")
            pass
        elif sample < self.p_empty + self.p_rand:  # 0-p_empty + p_rand>
            logger.warning("Taking Random Step")
            self.env.step(random.choice(self.env.storage_module.skills))
        else:  # The rest
            self.env.step(skill)

    def reset(self) -> tuple[EnvironmentObservation, EnvironmentObservation]:
        return self.env.reset()

    def evaluate(self) -> tuple[float, bool]:
        return self.env.evaluate()

    def close(self):
        self.env.close()
