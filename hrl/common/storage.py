from dataclasses import dataclass
from functools import cached_property
import os

from data.skills.skills import SKILLS_BY_TAG
from data.states.states import STATES_BY_TAG
from hrl.common.state import State
from hrl.common.skill import Skill


@dataclass
class StorageConfig:
    skills_tag: str
    states_tag: str
    storage_path: str = "data"
    results_path: str = "results"
    buffer_path: str = "logs"
    plots_path: str = "plots"


class StorageModule:
    def __init__(self, config: StorageConfig, nt: str, tag: str):
        self.config = config
        self.nt = nt
        self.tag = tag

    def create_directory(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def agent_saving_path(self) -> str:
        directory_path = self.config.storage_path + "/" + self.nt + "/" + self.tag + "/"
        return self.create_directory(directory_path)

    @cached_property
    def buffer_saving_path(self) -> str:
        directory_path = self.agent_saving_path + self.config.buffer_path + "/"
        return self.create_directory(directory_path)

    @cached_property
    def plots_saving_path(self) -> str:
        directory_path = self.agent_saving_path + self.config.plots_path + "/"
        return self.create_directory(directory_path)

    @cached_property
    def skills(self) -> list[Skill]:
        # We sort based on Id for the baseline network to be consistent
        skills = sorted(
            SKILLS_BY_TAG.get(self.config.skills_tag, []), key=lambda s: s.id
        )
        for skill in skills:
            skill.initialize_conditions(self.states)
            skill.initialize_overrides(self.states)
        return skills

    @cached_property
    def states(self) -> list[State]:
        # We sort based on Id for the baseline network to be consistent
        return sorted(STATES_BY_TAG.get(self.config.states_tag, []), key=lambda s: s.id)
