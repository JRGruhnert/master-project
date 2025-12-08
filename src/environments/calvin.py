from dataclasses import dataclass
from tapas_gmm.env.calvin import Calvin
from src.modules.evaluators.evaluator import Evaluator
from src.modules.storage import Storage
from src.environments.environment import Environment, EnvironmentConfig
from src.observation.observation import StateValueDict
from src.skills.empty import EmptySkill
from src.skills.skill import Skill

from src.observation.calvin import CalvinObservation
from src.skills.tapas import TapasSkill
from tapas_gmm.env.calvin import CalvinConfig


@dataclass
class CalvinEnvironmentConfig(EnvironmentConfig):
    calvin_config: CalvinConfig = CalvinConfig(
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
        eval_mode=False,
        real_time=False,
        pybullet_vis=False,
    )


class CalvinEnvironment(Environment):
    def __init__(
        self,
        config: CalvinEnvironmentConfig,
        evaluator: Evaluator,
        storage: Storage,
    ):
        self.config = config
        self.evaluator = evaluator
        self.storage = storage

        self.env = Calvin(self.config.calvin_config)
        self.info = {}

    def reset(self):
        temp = self.env.reset(settle_time=50)[0]
        self.goal = CalvinObservation.from_internal(temp)
        self.calvin_obs = self.env.reset(settle_time=50)[0]
        self.current = CalvinObservation.from_internal(self.calvin_obs)

    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        self.reset()
        # Current and goal should not be equal
        while not self.evaluator.is_valid_sample(self.current, self.goal):
            self.reset()
        return self.current, self.goal

    def step(
        self,
        skill: Skill,
    ) -> tuple[StateValueDict, float, bool]:
        assert isinstance(
            skill,
            TapasSkill,
        ) or isinstance(
            skill, EmptySkill
        ), "CalvinEnvironment only supports TapasSkill at this time."
        skill.reset(self.goal, self.env)
        while (action := skill.predict(self.calvin_obs)) is not None:
            self.calvin_obs = self.env.step(action, self.config.render, self.info)[0]
            self.current = CalvinObservation.from_internal(self.calvin_obs)

        reward, done = self.evaluator.step(self.current, self.goal)
        return self.current, reward, done

    def close(self):
        self.env.close()
