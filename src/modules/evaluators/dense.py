from dataclasses import dataclass
from src.modules.evaluators.evaluator import EvaluatorConfig, Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict


@dataclass
class DenseEvaluatorConfig(EvaluatorConfig):
    negative_step_reward: float
    positive_step_reward: float


class DenseEvaluator(Evaluator):
    def __init__(
        self,
        config: DenseEvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(storage)
        self.config = config

    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        prev_percentage_done = self.percentage_done
        if self.is_equal(current, goal):
            # Success reached
            return self.config.success_reward, True
        else:
            # Success not reached
            if prev_percentage_done < self.percentage_done:
                return self.config.positive_step_reward, False
            else:
                return self.config.negative_step_reward, False
