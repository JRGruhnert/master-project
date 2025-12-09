from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.states.calvin import AreaEulerState


@dataclass
class EvaluatorConfig:
    success_reward: float


class Evaluator(ABC):
    def __init__(
        self,
        storage: Storage,
    ):
        self.storage = storage
        self.percentage_done: float = 0.0

    def is_valid_sample(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Checks if the sampled states are valid for starting an episode."""
        is_done = self.is_equal(current, goal)
        is_good = self.is_good_sample(current, goal)
        return not is_done and is_good

    def is_equal(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Generic method to check if states match target conditions."""
        finished_states = 0
        for state in self.storage.eval_states:
            if state.name in current.keys():
                if state.evaluate(current[state.name], goal[state.name]):
                    finished_states += 1
        self.percentage_done = finished_states / max(len(self.storage.eval_states), 1)
        return finished_states == len(self.storage.eval_states)

    def is_good_sample(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        """Special method to check wether the sampled states are buggy or not."""
        # print(f"Checking states dense reward module...")
        for state in self.storage.states:
            if isinstance(state, AreaEulerState) and state.name in goal.keys():
                if not state.is_in_an_existing_area(goal[state.name]):
                    # print(f"Bad sample: goal {goal[state.name]} not in an area.")
                    return False
                if not state.is_in_an_existing_area(current[state.name]):
                    # print(f"Bad sample: current {current[state.name]} not in an area.")
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
