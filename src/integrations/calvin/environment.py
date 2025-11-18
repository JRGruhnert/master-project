from tapas_gmm.env.calvin import Calvin
import torch
from src.core.modules.reward_module import RewardModule
from src.core.modules.storage_module import StorageModule
from src.core.environment import BaseEnvironment, EnvironmentConfig
from src.core.skills.skill import BaseSkill

from src.integrations.calvin.observation import CalvinObservation
from src.integrations.calvin.config import calvin_config
from src.core.skills.tapas import TapasSkill


class CalvinEnvironment(BaseEnvironment):
    def __init__(
        self,
        config: EnvironmentConfig,
        reward_module: RewardModule,
        storage_module: StorageModule,
    ):
        self.config = config
        self.reward_module = reward_module
        self.storage_module = storage_module

        self.env = Calvin(calvin_config)

    def reset(self):
        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = CalvinObservation(goal_calvin)
        self.current_env, _, _, _ = self.env.reset(settle_time=50)
        self.current = CalvinObservation(self.current_env)

    def sample_task(self) -> tuple[CalvinObservation, CalvinObservation]:
        self.reset()
        # Current and goal should not be equal
        while self.reward_module.is_equal(self.current, self.goal):
            self.current_env, _, _, _ = self.env.reset(settle_time=50)
            self.current = CalvinObservation(self.current_env)

        return self.current, self.goal

    def is_good_sample(self):
        """This checks if the sampled Task is reasonable big."""
        pass

    def step(
        self,
        skill: BaseSkill,
    ) -> CalvinObservation:
        if isinstance(skill, TapasSkill):
            skill.reset(self.env)
            while (
                action := skill.predict(
                    self.current_env,
                    self.goal,
                )
            ) is not None:
                self.current_env, _, _, _ = self.env.step(action, self.config.render)
                self.current = CalvinObservation(self.current_env)
        else:
            raise NotImplementedError(
                "Only TapasSkill is implemented for CalvinEnvironment."
            )  # NOTE: This is not good. Need to make it generalize
        return self.current

    def close(self):
        self.env.close()

    def evaluate(self) -> tuple[float, bool]:
        return self.reward_module.step(self.current, self.goal)

    def evaluate_skill(self, values: dict[str, torch.Tensor]) -> bool:
        return self.reward_module.is_skill_equal(values, self.current)

    def sample_skill_prerequisites(
        self,
        skill: BaseSkill,
        prerequisites: tuple[str, torch.Tensor] | None = None,
    ) -> tuple[CalvinObservation, CalvinObservation]:
        self.reset()
        while True:
            self.current_env, _, _, _ = self.env.reset(settle_time=50)
            self.current = CalvinObservation(self.current_env)
            if self.evaluate_skill(skill.precons):
                break
        return self.current, self.goal
