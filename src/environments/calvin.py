from tapas_gmm.env.calvin import Calvin
import torch
from src.modules.evaluators.evaluator import Evaluator
from src.modules.storage import Storage
from src.environments.environment import Environment, EnvironmentConfig
from src.observation.observation import StateValueDict
from src.skills.skill import Skill

from src.observation.calvin import CalvinObservation
from src.skills.tapas import TapasSkill
from tapas_gmm.env.calvin import CalvinConfig


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

    def reset(self):
        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = CalvinObservation.from_internal(goal_calvin)
        self.current_env, _, _, _ = self.env.reset(settle_time=50)
        self.current = CalvinObservation.from_internal(self.current_env)

    def sample_task(self) -> tuple[StateValueDict, StateValueDict]:
        self.reset()
        # Current and goal should not be equal
        while self.evaluator.is_equal(self.current, self.goal):
            self.current_env, _, _, _ = self.env.reset(settle_time=50)
            self.current = CalvinObservation.from_internal(self.current_env)

        return self.current, self.goal

    def step(
        self,
        skill: Skill,
    ) -> tuple[StateValueDict, float, bool]:
        if isinstance(skill, TapasSkill):
            skill.reset(self.env)
            while (
                action := skill.predict(
                    self.current_env,
                    self.goal,
                )
            ) is not None:
                self.current_env, _, _, _ = self.env.step(action, self.config.render)
                self.current = CalvinObservation.from_internal(self.current_env)
        else:
            raise NotImplementedError(
                "Only TapasSkill is implemented for CalvinEnvironment."
            )  # NOTE: This is not good. Need to make it generalize
        reward, done = self.evaluator.step(self.current, self.goal)
        return self.current, reward, done

    def close(self):
        self.env.close()

    def evaluate_skill(self, values: dict[str, torch.Tensor]) -> bool:
        return self.evaluator.is_skill_equal(
            StateValueDict.from_tensor_dict(values), self.current
        )

    def sample_for_skill(
        self,
        skill: Skill,
        prerequisites: tuple[str, torch.Tensor] | None = None,
    ) -> tuple[StateValueDict, StateValueDict]:
        self.reset()
        while True:
            self.current_env, _, _, _ = self.env.reset(settle_time=50)
            self.current = CalvinObservation.from_internal(self.current_env)
            if self.evaluate_skill(skill.precons):
                break
        return self.current, self.goal
