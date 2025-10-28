from dataclasses import dataclass
from functools import cached_property
import os

from src.integrations.tapas.skills import SKILLS_BY_TAG
from src.integrations.calvin.states import STATES_BY_TAG
from src.core.state import BaseState
from src.core.skills.skill import BaseSkill
from src.core.networks import NetworkType


@dataclass
class StorageConfig:
    skills_tag: str
    states_tag: str
    storage_path: str = "data"
    results_path: str = "results"
    buffer_path: str = "logs"
    plots_path: str = "plots"
    checkpoint_path: str = "checkpoint.pth"


class StorageModule:
    def __init__(
        self,
        config: StorageConfig,
        tag: str,
        nt: NetworkType,
    ):
        self.config = config
        self.tag = tag
        self.nt = nt

    def create_directory(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @cached_property
    def agent_saving_path(self) -> str:
        directory_path = (
            self.config.results_path + "/" + self.nt.value + "/" + self.tag + "/"
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

    @cached_property
    def skills(self) -> list[BaseSkill]:
        # We sort based on Id for the baseline network to be consistent
        skills = sorted(
            SKILLS_BY_TAG.get(self.config.skills_tag, []), key=lambda s: s.id
        )
        print(
            f"Loaded skills for tag {self.config.skills_tag}: {[s.name for s in skills]}"
        )
        for skill in skills:
            skill.initialize_conditions(self.states)
            skill.initialize_overrides(self.states)
        return skills

    @cached_property
    def states(self) -> list[BaseState]:
        # We sort based on Id for the baseline network to be consistent
        return sorted(STATES_BY_TAG.get(self.config.states_tag, []), key=lambda s: s.id)
