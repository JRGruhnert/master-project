from dataclasses import dataclass
from src.modules.evaluators.evaluator import EvaluatorConfig, Evaluator
from src.modules.storage import Storage
from src.observation.observation import StateValueDict


@dataclass
class Dense3EvaluatorConfig(EvaluatorConfig):
    max_progress_reward: float = 1.0
    # Small step penalty to encourage efficiency
    step_penalty: float = -0.002
    add_monotonic_reward: bool = True
    success_reward: float = 25.0


class Dense3Evaluator(Evaluator):
    def __init__(
        self,
        config: Dense3EvaluatorConfig,
        storage: Storage,
    ):
        super().__init__(storage)
        self.config = config
        self.max_percentage_done: float = 0.0

    def is_valid_sample(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> bool:
        valid = super().is_valid_sample(current, goal)
        self.max_percentage_done = max(self.max_percentage_done, self.percentage_done)
        # print("Resetting")
        return valid

    def step(
        self,
        current: StateValueDict,
        goal: StateValueDict,
    ) -> tuple[float, bool]:
        prev_percentage_done = self.percentage_done

        if self.is_equal(current, goal):
            return self.config.success_reward, True

        improvement = self.percentage_done - prev_percentage_done
        # print("improvement:", improvement)
        reward = max(0.0, improvement * self.config.max_progress_reward)
        # print("reward:", reward)
        if self.config.add_monotonic_reward:
            # print("max percentage done:", self.max_percentage_done)
            # print("current percentage done:", self.percentage_done)
            if self.percentage_done > self.max_percentage_done:
                reward += (
                    self.percentage_done - self.max_percentage_done
                ) * self.config.max_progress_reward
                self.max_percentage_done = self.percentage_done
        # Add small step penalty
        reward += self.config.step_penalty
        # print("reward after step penalty:", reward)
        return reward, False
