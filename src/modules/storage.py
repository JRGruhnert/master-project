from dataclasses import dataclass
from functools import cached_property
import os

from src.skills.skill import Skill
from src.skills.skills import SKILLS_BY_TAG
from src.skills.tapas import TapasSkill
from src.states.state import State
from src.states.states import STATES_BY_TAG


@dataclass
class StorageConfig:
    skills_tag: str
    states_tag: str
    tag: str
    network: str
    storage_path: str = "data"
    results_path: str = "results"
    buffer_path: str = "logs"
    plots_path: str = "plots"
    checkpoint_path: str = "checkpoint.pth"


class Storage:
    def __init__(
        self,
        config: StorageConfig,
    ):
        self.config = config
        self.states: list[State] = sorted(
            STATES_BY_TAG.get(self.config.states_tag, []), key=lambda s: s.id
        )
        # We sort based on Id for the baseline network to be consistent
        self.skills: list[Skill] = sorted(
            SKILLS_BY_TAG.get(self.config.skills_tag, []), key=lambda s: s.id
        )
        print(
            f"Loaded skills for tag {self.config.skills_tag}: {[s.name for s in self.skills]}"
        )
        for skill in self.skills:
            if isinstance(skill, TapasSkill):
                skill.initialize_conditions(self.states)
                skill.initialize_overrides(self.states)

    def create_directory(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def agent_saving_path(self) -> str:
        directory_path = (
            self.config.results_path
            + "/"
            + self.config.network
            + "/"
            + self.config.tag
            + "/"
        )
        return self.create_directory(directory_path)

    @cached_property
    def buffer_saving_path(self) -> str:
        directory_path = self.agent_saving_path + self.config.buffer_path + "/"
        return self.create_directory(directory_path)

    @cached_property
    def plots_saving_path(self) -> str:
        directory_path = self.agent_saving_path + self.config.plots_path + "/"
        return self.create_directory(directory_path)
