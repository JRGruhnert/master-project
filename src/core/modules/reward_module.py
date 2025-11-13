from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.observation import BaseObservation
from src.core.skills.skill import BaseSkill
from src.core.state import BaseState


@dataclass
class RewardConfig:
    step_reward: float
    success_reward: float


class RewardModule(ABC):
    def __init__(self, config: RewardConfig, states: list[BaseState]):
        self.states = states
        self.config = config
        self.first_step = True
        self.percentage = 0.0

    def _check_states(
        self, current: BaseObservation | dict, goal: BaseObservation
    ) -> tuple[float, bool]:
        """Generic method to check if states match target conditions."""
        # print(f"Checking states dense reward module...")
        not_finished_states = 0
        for state in self.states:
            if state.name in current.keys():
                if not state.evaluate(current[state.name], goal[state.name]):
                    print(f"Wrong: \t {state.name}")
                    print(f"{current[state.name]} is not {goal[state.name]}")
                    not_finished_states += 1
        return not_finished_states / max(len(self.states), 1), not_finished_states == 0

    @abstractmethod
    def step(
        self, current: BaseObservation, goal: BaseObservation
    ) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        raise NotImplementedError()

    def is_equal(self, current: BaseObservation, goal: BaseObservation) -> bool:
        # print(f"RewardModule is_equal check...")
        self.percentage, done = self._check_states(current, goal)
        return done

    def is_skill_start(
        self, skill: BaseSkill, current: BaseObservation
    ) -> tuple[float, bool]:
        _, done = self._check_states(skill.precons, current)
        if done:
            return self.config.success_reward, True
        else:
            return self.config.step_reward, False

    def is_skill_end(
        self, skill: BaseSkill, current: BaseObservation
    ) -> tuple[float, bool]:
        _, done = self._check_states(skill.postcons, current)
        if done:
            return self.config.success_reward, True
        else:
            return self.config.step_reward, False

    def distance_to_skill(
        self,
        current: BaseObservation,
        goal: BaseObservation,
        skill: BaseSkill,
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
        current: BaseObservation,
        goal: BaseObservation,
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


class SparseRewardModule(RewardModule):

    def step(
        self, current: BaseObservation, goal: BaseObservation
    ) -> tuple[float, bool]:
        # print(f"RewardModule step check...")
        self.percentage, done = self._check_states(current, goal)
        if done:
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            return self.config.step_reward, False


class DenseRewardModule(RewardModule):

    def step(
        self, current: BaseObservation, goal: BaseObservation
    ) -> tuple[float, bool]:
        # print(f"RewardModule step check...")
        old_percentage = self.percentage
        self.percentage, done = self._check_states(current, goal)
        if done:
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            progress_reward = (
                self.config.success_reward * 0.01 * (self.percentage - old_percentage)
            )
            return progress_reward, False
