from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.skills.skill import Skill
from src.states.calvin import AreaEulerState
from src.states.state import State


@dataclass
class EvaluatorConfig:
    success_reward: float


class Evaluator(ABC):
    def __init__(
        self,
        storage: Storage,
    ):
        self.storage = storage
        self.states = storage.states
        self.first_step = True
        self.non_equal_states = 0.0

    def is_equal(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Generic method to check if states match target conditions."""
        if current.keys() != goal.keys():
            for key, value in current.items():
                for state in self.states:
                    if state.name == key:
                        if not state.evaluate(current[state.name], goal[state.name]):
                            print(f"{key} {value} is not {goal[key]}")
        not_finished_states = 0
        for state in self.states:
            if state.name in current.keys():
                if not state.evaluate(current[state.name], goal[state.name]):
                    # print(f"Wrong: \t {state.name}")
                    # print(f"{current[state.name]} is not {goal[state.name]}")
                    not_finished_states += 1
        self.non_equal_states = not_finished_states / max(len(self.states), 1)
        return self.non_equal_states == 0.0

    def is_good_sample(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Special method to check wether the sampled states are buggy or not."""
        # print(f"Checking states dense reward module...")
        for state in self.states:
            if isinstance(state, AreaEulerState) and state.name in goal.keys():
                if not state.is_in_area(goal[state.name]):
                    print(f"Bad sample: goal {goal[state.name]} not in an area.")
                    return False
                if not state.is_in_area(current[state.name]):
                    print(f"Bad sample: current {current[state.name]} not in an area.")
                    return False
        return True

    def same_areas(self, conditions: StateValueDict, goal: StateValueDict) -> bool:
        """Checks if all area states are equal between conditions and goal."""
        for key, value in conditions.items():
            state = self.storage.get_state_by_name(key)
            if isinstance(state, AreaEulerState):
                if not state.evaluate(value, goal[key]):
                    return False
        return True

    @abstractmethod
    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        raise NotImplementedError()
