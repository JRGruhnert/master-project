from dataclasses import dataclass
from src.integrations.calvin.environment import CalvinEnvironment
from src.integrations.calvin.observation import CalvinObservation


@dataclass
class RetrainConfig:
    checkpoint: str = ""


class RetrainExperiment:
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: RetrainConfig, env: CalvinEnvironment):
        # We sort based on Id for the baseline network to be consistent

        self.checkpoint = config.checkpoint
        self.env = env

    def step(self, skill) -> tuple[CalvinObservation, float, bool]:
        return self.env.step(skill)

    def reset(self) -> tuple[CalvinObservation, CalvinObservation]:
        return self.env.reset()

    def evaluate(self) -> tuple[float, bool]:
        return self.env.evaluate()

    def close(self):
        self.env.close()
