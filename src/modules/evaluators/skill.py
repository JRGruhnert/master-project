from dataclasses import dataclass

from src.modules.evaluators.evaluator import Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.states.calvin import AreaEulerState


@dataclass
class SkillEvaluatorConfig:
    pass


class SkillEvaluator(Evaluator):
    """Evaluator for skill execution based on skill preconditions and postconditions.
    NOTE: This evaluator is specific to the SkillCheckExperiment and Tapas Skills.
    """

    def __init__(
        self,
        config: SkillEvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(storage)
        self.config = config

    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        "Returns the step reward and wether the step is a terminal step, cause some ending condition was met."
        return 0.0, self.is_equal(current, goal)

    def same_areas(self, conditions: StateValueDict, goal: StateValueDict) -> bool:
        """Checks if all area states are equal between conditions and goal."""
        for key, value in conditions.items():
            state = self.storage.get_state_by_name(key)
            if isinstance(state, AreaEulerState):
                if not state.evaluate(value, goal[key]):
                    return False
        return True
