from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.core.observation import BaseObservation
from src.core.skill import BaseSkill
from src.core.state import BaseState


@dataclass
class RewardConfig:
    step_reward: float
    success_reward: float


class RewardModule(ABC):
    def __init__(self, config: RewardConfig, states: list[BaseState]):
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
        # print(f"Checking states sparse reward module...")
        finished = True
        for state in self.states:
            if state.name in goal:
                if not state.evaluate(current[state.name], goal[state.name]):
                    print(
                        f"NOPE: State {state.name} {current[state.name]} {goal[state.name]}"
                    )
                    finished = False
        return finished

    def step(
        self, current: BaseObservation, goal: BaseObservation
    ) -> tuple[float, bool]:
        print(f"RewardModule step check...")
        if self._check_states(
            current.top_level_observation, goal.top_level_observation
        ):
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            return self.config.step_reward, False

    def is_skill_start(self, skill: BaseSkill, current: BaseObservation) -> bool:
        return self._check_states(skill.precons, current.top_level_observation)

    def is_skill_end(self, skill: BaseSkill, current: BaseObservation) -> bool:
        return self._check_states(skill.postcons, current.top_level_observation)

    def is_equal(self, current: BaseObservation, goal: BaseObservation) -> bool:
        print(f"RewardModule is_equal check...")
        return self._check_states(
            current.top_level_observation, goal.top_level_observation
        )

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
                    current.top_level_observation[state.name],
                    goal.top_level_observation[state.name],
                    skill.precons[state.name],
                )
                total_distance += distance
        return total_distance / len(skill.precons)

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
        return total_distance / len(self.states)
