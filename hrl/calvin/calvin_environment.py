from tapas_gmm.env.calvin import Calvin, CalvinConfig
from hrl.common.modules.reward_module import SparseRewardModule
from hrl.common.modules.storage_module import StorageModule
from hrl.common.environment import BaseEnvironment, EnvironmentConfig
from hrl.calvin.calvin_observation import CalvinObservation
from hrl.common.skill import BaseSkill


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

        # Calvin specific
        c_config = CalvinConfig(
            task="Undefined",
            cameras=("wrist", "front"),
            camera_pose={},
            image_size=(256, 256),
            static=False,
            headless=False,
            scale_action=False,
            delay_gripper=False,
            gripper_plot=False,
            postprocess_actions=False,
            eval_mode=True,
            pybullet_vis=False,
            real_time=False,
        )
        self.env = Calvin(c_config)

    def reset(
        self, skill: BaseSkill = None
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
    ):
        skill.reset(
            self.env,
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
            self.current_env, _, _, _ = self.env.step(action, self.config.render)
            self.current = CalvinObservation(self.current_env)

    def close(self):
        self.env.close()

    def evaluate(self, skill: BaseSkill = None) -> tuple[float, bool]:
        if skill:
            return self.reward_module.is_skill_end(skill, self.current)
        else:
            return self.reward_module.step(self.current, self.goal)
