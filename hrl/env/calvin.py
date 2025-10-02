from dataclasses import dataclass
from enum import Enum
import random
from loguru import logger

from tapas_gmm.env.calvin import Calvin, CalvinConfig
from hrl.env.observation import EnvironmentObservation
from hrl.skill.tapas import Tapas
from hrl.skill.skill import Skill
from hrl.state.state import State, TapasState


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


@dataclass
class MasterEnvConfig:
    debug_vis: bool = False
    # Reward Settings
    reward_mode: RewardMode = RewardMode.SPARSE
    max_reward: float = 100.0
    min_reward: float = -1.0


class CalvinEnvironment:
    def __init__(
        self,
        config: MasterEnvConfig,
        max_steps: int,
    ):
        self.config = config
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
        self.skill: Skill = None
        self.max_steps, self.steps_left = max_steps, max_steps  # Cached property
        self.terminal = False

    def reset(
        self, states: list[State], skill: Skill = None
    ) -> tuple[EnvironmentObservation, EnvironmentObservation]:

        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = EnvironmentObservation(goal_calvin)

        self.current_env, _, _, _ = self.env.reset(settle_time=50)
        self.current = EnvironmentObservation(self.current_env)
        if skill:
            while not self.startposition_check(skill, states):
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = EnvironmentObservation(self.current_env)
        else:
            while self.completion_check(states):  # Ensure that they are not the same
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = EnvironmentObservation(self.current_env)

        self.steps_left = self.max_steps
        self.terminal = False
        return self.current, self.goal

    def step_exp1(
        self,
        skill: Skill,
        skills: list[Skill],
        states: list[State],
        p_empty: float = 0.0,
        p_rand: float = 0.0,
    ) -> tuple[float, bool, EnvironmentObservation]:
        sample = random.random()
        if sample < p_empty:  # 0-p_empty>
            logger.warning("Taking Empty Step")
            pass
        elif sample < p_empty + p_rand:  # 0-p_empty + p_rand>
            logger.warning("Taking Random Step")
            self.step(random.choice(skills))
        else:  # The rest
            self.step(skill, states)
        self.steps_left -= 1
        reward, done = self.evaluate(states)
        return reward, done, self.current

    def step(
        self,
        skill: Tapas,
        states: list[TapasState],
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
                action := skill.predict(self.current_env, self.goal, states)
            ) is not None:
                self.current_env, _, _, _ = self.env.step(action, self.config.debug_vis)
                self.current = EnvironmentObservation(self.current_env)
        except Exception as e:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            logger.error(f"Error happened during step: {e}")

    def close(self):
        self.env.close()

    def evaluate(self, states: list[State], skill: Skill = None) -> tuple[float, bool]:
        if self.terminal:
            raise UserWarning(
                "Episode already ended. Please reset the evaluator with the new goal and state."
            )
        if self.config.reward_mode is RewardMode.SPARSE:
            if skill:
                if self.endposition_check(skill, states):
                    return self.config.max_reward, True
                else:
                    return self.config.min_reward, False
            if self.completion_check(states):
                return self.config.max_reward, True
            else:
                return self.config.min_reward, False if self.steps_left > 0 else True
        if self.config.reward_mode is RewardMode.ONOFF:
            raise NotImplementedError("Reward Mode not implemented.")
        if self.config.reward_mode is RewardMode.RANGE:
            raise NotImplementedError("Reward Mode not implemented.")

    def _check_states(
        self, target_states: dict, current_states: dict, states: list[State]
    ) -> bool:
        """Generic method to check if states match target conditions."""
        for state in states:
            if state.name in target_states:
                if not state.evaluate(
                    current_states[state.name], target_states[state.name]
                ):
                    print(f"State {state.name} does not match")
                    return False
                print(f"State {state.name} not in target states")
        return True

    def startposition_check(self, skill: Skill, states: list[State]) -> bool:
        """Checking if skill start position is reached"""
        return self._check_states(skill.precons, self.current.states, states)

    def endposition_check(self, skill: Skill, states: list[State]) -> bool:
        """Checking if skill end position is reached"""
        return self._check_states(skill.postcons, self.current.states, states)

    def completion_check(self, states: list[State]) -> bool:
        """Checking if goal is reached"""
        return self._check_states(self.goal.states, self.current.states, states)
