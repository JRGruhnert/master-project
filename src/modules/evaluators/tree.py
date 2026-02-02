from dataclasses import dataclass
from src.modules.evaluators.dense3 import Dense3EvaluatorConfig, Dense3Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.skills.skill import Skill


@dataclass
class TreeEvaluatorConfig(Dense3EvaluatorConfig):
    pass


class TreeEvaluator(Dense3Evaluator):
    def __init__(
        self,
        config: TreeEvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(config, storage)

    def distance_to_skill(
        self,
        current: StateValueDict,
        skill: Skill,
    ) -> float:
        """Returns a distance metric indicating how far the current observation is from satisfying the skill's preconditions."""
        total_distance = 0.0
        for state in self.storage.eval_states:
            if state.name in skill.precons:
                distance = state.distance_to_skill(
                    current[state.name],
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
        for state in self.storage.eval_states:
            distance = state.distance_to_goal(
                current[state.name],
                goal[state.name],
            )
            total_distance += distance
        return total_distance / max(len(self.storage.eval_states), 1)
