from dataclasses import dataclass
from src.modules.evaluators.evaluator import EvaluatorConfig, Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict


@dataclass
class DenseBestEvaluatorConfig(EvaluatorConfig):
    positive_step_reward: float
    negative_step_reward: float


class Dense2Evaluator(Evaluator):
    def __init__(
        self,
        config: DenseBestEvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(storage)
        self.config = config
        self.starting_non_equal_states = 0.0

    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        if self.first_step:
            self.first_step = False
            self.starting_non_equal_states = self.non_equal_states

        if self.is_equal(current, goal):
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            if self.starting_non_equal_states > self.non_equal_states:
                return self.config.positive_step_reward, False
            else:
                return self.config.negative_step_reward, False
