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
    used_skills: str
    used_states: str
    network: str
    tag: str = "untagged_run"
    storage_path: str = "data"
    results_path: str = "results"
    buffer_path: str = "logs"
    plots_path: str = "plots"
    checkpoint_path: str = "checkpoint.pth"
    eval_states: str | None = None


class Storage:
    def __init__(
        self,
        config: StorageConfig,
    ):
        self.config = config
        self._states: list[State] = sorted(
            STATES_BY_TAG.get(self.config.used_states, []), key=lambda s: s.id
        )

        # Evaluation states can be different from training states
        # Just means that the relevant states for sampling and evaluation are different
        if self.config.eval_states is not None:
            self._eval_states = sorted(
                STATES_BY_TAG.get(self.config.eval_states, []), key=lambda s: s.id
            )
        else:
            self._eval_states = self.states
        # We sort based on Id for the baseline network to be consistent
        self._skills: list[Skill] = sorted(
            SKILLS_BY_TAG.get(self.config.used_skills, []), key=lambda s: s.id
        )
        print(
            f"Loaded skills for tag {self.config.used_skills}: {[s.name for s in self.skills]}"
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

    def get_skill_by_name(self, name: str) -> Skill | None:
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def get_state_by_name(self, name: str) -> State | None:
        for state in self.states:
            if state.name == name:
                return state
        return None

    @property
    def states(self) -> list[State]:
        return self._states

    @property
    def eval_states(self) -> list[State]:
        return self._eval_states

    @property
    def skills(self) -> list[Skill]:
        return self._skills
