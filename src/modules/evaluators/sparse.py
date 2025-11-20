from dataclasses import dataclass
from src.modules.evaluators.evaluator import EvaluatorConfig, Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict


@dataclass
class SparseEvaluatorConfig(EvaluatorConfig):
    step_reward: float


class SparseEvaluator(Evaluator):
    def __init__(
        self,
        config: SparseEvaluatorConfig,
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
