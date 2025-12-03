from dataclasses import dataclass
from src.modules.evaluators.evaluator import EvaluatorConfig, Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict


@dataclass
class Dense2EvaluatorConfig(EvaluatorConfig):
    # Reward for going from 0% correct to 100% correct (excluding success bonus)
    max_progress_reward: float = 0.5
    # Penalty for going from 100% correct to 0% correct
    max_regress_penalty: float = 0.25
    # Small step penalty to encourage efficiency
    step_penalty: float = -0.01


class Dense2Evaluator(Evaluator):
    def __init__(
        self,
        config: Dense2EvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(storage)
        self.config = config
        self.prev_percentage_done: float | None = None

    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        prev_percentage_done = self.percentage_done
        if self.is_equal(current, goal):
            # Success! Big reward
            return self.config.success_reward, True

        # Calculate change in correctness (-1 to +1)
        improvement = self.percentage_done - prev_percentage_done

        if improvement > 0:
            # Made progress - reward proportional to fraction improved
            reward = improvement * self.config.max_progress_reward
        elif improvement < 0:
            # Regressed - penalty proportional to fraction broken
            reward = (
                improvement * self.config.max_regress_penalty
            )  # improvement is negative
        else:
            # No change
            reward = 0.0

        # Add small step penalty
        reward += self.config.step_penalty

        return reward, False
