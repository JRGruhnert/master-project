from dataclasses import dataclass
import random

from loguru import logger
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation


@dataclass
class BeltConfig:
    p_empty: float
    p_rand: float


class TacoRLExperiment:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: BeltConfig, env: CalvinEnvironment):
        self.env = env

    def step(self, skill) -> tuple[CalvinObservation, float, bool]:
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
