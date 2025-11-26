from dataclasses import dataclass
from src.modules.evaluators.evaluator import EvaluatorConfig, Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


@dataclass
class TreeEvaluatorConfig(EvaluatorConfig):
    step_reward: float


class TreeEvaluator(Evaluator):
    def __init__(
        self,
        config: TreeEvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(storage)
        self.config = config

    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        if self.is_equal(current, goal):
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            return self.config.step_reward, False

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
                ).item()
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
            ).item()
            total_distance += distance
        return total_distance / max(len(self.states), 1)
