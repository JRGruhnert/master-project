from typing import Optional
from tapas_gmm.env.calvin import Calvin
from src.core.modules.reward_module import SparseRewardModule
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
        reward_module: SparseRewardModule,
        storage_module: StorageModule,
    ):
        self.config = config
        self.reward_module = reward_module
        self.storage_module = storage_module

        self.env = Calvin(calvin_config)

    def reset(
        self, skill: Optional[BaseSkill] = None
    ) -> tuple[CalvinObservation, CalvinObservation]:
        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = CalvinObservation(goal_calvin)

        self.current_env, _, _, _ = self.env.reset(settle_time=50)
        self.current = CalvinObservation(self.current_env)
        if skill:
            # Should fullfill preconditions for skill
            while not self.reward_module.is_skill_start(skill, self.current):
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = CalvinObservation(self.current_env)
        else:
            # Current and goal should not be equal
            while self.reward_module.is_equal(self.current, self.goal):
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = CalvinObservation(self.current_env)

        return self.current, self.goal

    def step(
        self,
        skill: BaseSkill,
        predict_as_batch: bool = True,
        control_duration: int = -1,
    ) -> CalvinObservation:
        if isinstance(skill, TapasSkill):
            skill.reset(
                self.env,
                predict_as_batch=predict_as_batch,
                control_duration=control_duration,
            )
        else:
            skill.reset(
                predict_as_batch=predict_as_batch,
                control_duration=control_duration,
            )
        # try:
        while (
            action := skill.predict(
                self.current_env,
                self.goal,
                self.storage_module.states,
            )
        ) is not None:
            # print(f"Action: {action[0:3]}")
            self.current_env, _, _, _ = self.env.step(action, self.config.render)
            # print(f"EE 1: {self.current['ee_position']}")
            self.current = CalvinObservation(self.current_env)
            # print(f"EE 2: {self.current['ee_position']}")

        return self.current

    def close(self):
        self.env.close()

    def evaluate(self, skill: Optional[BaseSkill] = None) -> tuple[float, bool]:
        if skill:
            return self.reward_module.is_skill_end(skill, self.current)
        else:
            return self.reward_module.step(self.current, self.goal)
