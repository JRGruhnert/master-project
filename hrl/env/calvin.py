from loguru import logger

from tapas_gmm.env.calvin import Calvin, CalvinConfig
from hrl.common.reward import SparseEval
from hrl.common.storage import Storage
from hrl.env.environment import BaseEnvironment, EnvironmentConfig
from hrl.env.observation import EnvironmentObservation
from hrl.common.skill import Skill


class CalvinEnvironment(BaseEnvironment):
    def __init__(
        self,
        config: EnvironmentConfig,
        eval: SparseEval,
        storage: Storage,
    ):
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
        self.config = config
        self.eval = eval
        self.storage = storage

    def reset(
        self, skill: Skill = None
    ) -> tuple[EnvironmentObservation, EnvironmentObservation]:
        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = EnvironmentObservation(goal_calvin)

        self.current_env, _, _, _ = self.env.reset(settle_time=50)
        self.current = EnvironmentObservation(self.current_env)
        if skill:
            # Should fullfill preconditions for skill
            while not self.eval.is_skill_start(skill, self.current):
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = EnvironmentObservation(self.current_env)
        else:
            # Current and goal should not be equal
            while self.eval.is_equal(self.current, self.goal):
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = EnvironmentObservation(self.current_env)

        return self.current, self.goal

    def step(
        self,
        skill: Skill,
        predict_as_batch: bool = True,
        control_duration: int = -1,
    ):
        skill.reset(
            self.env,
            predict_as_batch=predict_as_batch,
            control_duration=control_duration,
        )
        try:
            while (
                action := skill.predict(
                    self.current_env,
                    self.goal,
                    self.storage.states,
                )
            ) is not None:
                self.current_env, _, _, _ = self.env.step(action, self.config.render)
                self.current = EnvironmentObservation(self.current_env)
        except Exception as e:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            logger.error(f"Error happened during step: {e}")

    def close(self):
        self.env.close()

    def evaluate(self, skill: Skill = None) -> tuple[float, bool]:
        if skill:
            return self.eval.is_skill_end(skill, self.current)
        else:
            return self.eval.step(self.current, self.goal)
