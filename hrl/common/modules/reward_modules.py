from abc import ABC, abstractmethod
from dataclasses import dataclass

from hrl.env.observation import BaseObservation
from hrl.common.skill import Skill
from hrl.common.state import State


@dataclass
class RewardConfig:
    step_reward: float
    success_reward: float


class RewardModule(ABC):
    def __init__(self, config: RewardConfig, states: list[State]):
        self.states = states
        self.config = config

    @abstractmethod
    def step(
        self, current: BaseObservation, goal: BaseObservation
    ) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        raise NotImplementedError()

    @abstractmethod
    def is_equal(self, current: BaseObservation, goal: BaseObservation) -> bool:
        raise NotImplementedError()


class SparseRewardModule(RewardModule):

    def _check_states(self, current: dict, goal: dict) -> bool:
        """Generic method to check if states match target conditions."""
        for state in self.states:
            if state.name in goal:
                if not state.evaluate(current[state.name], goal[state.name]):
                    return False
        return True

    def step(
        self, current: BaseObservation, goal: BaseObservation
    ) -> tuple[float, bool]:
        if self._check_states(
            current.top_level_observation, goal.top_level_observation
        ):
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            return self.config.step_reward, False

    def is_skill_start(self, skill: Skill, current: BaseObservation) -> bool:
        return self._check_states(skill.precons, current.top_level_observation)

    def is_skill_end(self, skill: Skill, current: BaseObservation) -> bool:
        return self._check_states(skill.postcons, current.top_level_observation)

    def is_equal(self, current: BaseObservation, goal: BaseObservation) -> bool:
        return self._check_states(
            current.top_level_observation, goal.top_level_observation
        )
