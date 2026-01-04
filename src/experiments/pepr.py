from dataclasses import dataclass
import math
import random

from loguru import logger
from src.modules.storage import Storage
from src.skills.skill import Skill
from src.skills.empty import EmptySkill
from src.experiments.experiment import Experiment, ExperimentConfig
from src.environments.environment import Environment
from src.observation.observation import StateValueDict


@dataclass
class PePrConfig(ExperimentConfig):
    p_empty: float
    p_rand: float


class PePrExperiment(Experiment):
    """Simple Wrapper for centralized data loading and initialisation."""

    def __init__(self, config: PePrConfig, env: Environment, storage: Storage):
        # We sort based on Id for the baseline network to be consistent
        super().__init__(config, env, storage)
        self.config = config
        if storage.config.used_skills == "b":
            num_skills = 6
        elif (
            storage.config.used_skills == "sr"
            or storage.config.used_skills == "sb"
            or storage.config.used_skills == "sp"
        ):
            num_skills = 8
        elif (
            storage.config.used_skills == "br"
            or storage.config.used_skills == "bb"
            or storage.config.used_skills == "bp"
        ):
            num_skills = 10
        elif storage.config.used_skills == "brb":
            num_skills = 12
        elif storage.config.used_skills == "brbp":
            num_skills = 14
        # NOTE: This is for my skills setup
        self.max_episode_length = math.ceil(
            num_skills
            + num_skills * self.config.p_empty
            + num_skills * self.config.p_rand
        )

        self.current_step = 0
        self.current: StateValueDict | None = None
        logger.info(
            f"Number of skills: {num_skills}, max episode length: {self.max_episode_length}"
        )

    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        self.current_step = 0
        self.current, goal = self.env.sample_task()
        return self.current, goal

    def step(self, skill: Skill) -> tuple[StateValueDict, float, bool, bool]:
        self.current_step += 1
        sample = random.random()
        if sample < self.config.p_empty:  # 0-p_empty>
            logger.info("Taking Empty Step")
            overwrite_skill = EmptySkill()
        elif sample < self.config.p_empty + self.config.p_rand:  # 0-p_empty + p_rand>
            logger.info("Taking Random Step")
            overwrite_skill = random.choice(self.storage.skills)
        else:  # The rest
            overwrite_skill = skill

        self.current, reward, done = self.env.step(overwrite_skill)
        terminal = True if self.current_step >= self.max_episode_length else done
        return self.current, reward, done, terminal

    def metadata(self) -> dict:
        return {
            "p_empty": self.config.p_empty,
            "p_rand": self.config.p_rand,
            "max_episode_length": self.max_episode_length,
        }
