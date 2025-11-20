from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.skills.skill import Skill
from src.states.state import State


@dataclass
class EvaluatorConfig:
    success_reward: float


class Evaluator(ABC):
    def __init__(
        self,
        storage: Storage,
    ):
        self.states = storage.states
        self.first_step = True
        self.non_equal_states = 0.0

    def is_equal(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Generic method to check if states match target conditions."""
        # print(f"Checking states dense reward module...")
        not_finished_states = 0
        for state in self.states:
            if state.name in current.keys():
                if not state.evaluate(current[state.name], goal[state.name]):
                    # print(f"Wrong: \t {state.name}")
                    # print(f"{current[state.name]} is not {goal[state.name]}")
                    not_finished_states += 1
        self.non_equal_states = not_finished_states / max(len(self.states), 1)
        return self.non_equal_states == 0.0

    @abstractmethod
    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        raise NotImplementedError()

    def distance_to_skill(
        self,
        current: StateValueDict,
        goal: StateValueDict,
        skill: Skill,
    ) -> float:
        """Returns a distance metric indicating how far the current observation is from satisfying the skill's preconditions."""
        total_distance = 0.0
        for state in self.states:
            if state.name in skill.precons:
                distance = state.distance_to_skill(
                    current[state.name],
                    goal[state.name],
                    skill.precons[state.name],
                )
                total_distance += distance
        return total_distance / max(len(skill.precons), 1)

    def distance_to_goal(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> float:
        """Generic method to check if states match target conditions."""
        # print(f"Checking states sparse reward module...")
        total_distance = 0.0
        for state in self.states:
            distance = state.distance_to_goal(
                current[state.name],
                goal[state.name],
            )
            total_distance += distance
        return total_distance / max(len(self.states), 1)

    def is_skill_equal(
        self, skill_states: StateValueDict, current: StateValueDict
    ) -> bool:
        values = skill_states.copy()  # Python passes by reference so..
        # NOTE: Again an exception for the Flip State...
        if "base__button_scalar" in values:
            values["base__button_scalar"] = current["base__button_scalar"]
        return self.is_equal(values, current)
