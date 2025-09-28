from dataclasses import dataclass
from enum import Enum
import random
from loguru import logger
import numpy as np
import re
import torch

from calvin_env.envs.observation import CalvinObservation
from tapas_gmm.env.calvin import Calvin, CalvinConfig
from tapas_gmm.utils.observation import (
    CameraOrder,
    SceneObservation,
    SingleCamObservation,
    dict_to_tensordict,
    empty_batchsize,
)
from hrl.state.state import State
from hrl.skill.skill import Skill
from hrl.observation.observation import MPObservation


class RewardMode(Enum):
    SPARSE = 0
    RANGE = 1
    ONOFF = 2


@dataclass
class MasterEnvConfig:
    calvin_config: CalvinConfig
    debug_vis: bool
    # Reward Settings
    reward_mode: RewardMode = RewardMode.SPARSE
    max_reward: float = 100.0
    min_reward: float = -1.0


class MasterEnv:
    def __init__(
        self,
        config: MasterEnvConfig,
        states: list[State],
        tasks: list[Skill],
        max_steps: int,
    ):
        self.config = config
        self.states = states
        self.tasks = tasks
        self.env = Calvin(config=config.calvin_config)

        self.last_gripper_action = [1.0]  # open
        self.max_steps, self.steps_left = max_steps, max_steps  # Cached property
        self.terminal = False
        self.task: Skill = None

    def reset(self, task: Skill = None) -> tuple[MPObservation, MPObservation]:

        goal_calvin, _, _, _ = self.env.reset(settle_time=50)
        self.goal = MPObservation(goal_calvin)

        self.current_env, _, _, _ = self.env.reset(settle_time=50)
        self.current = MPObservation(self.current_env)
        if task:
            self.task = task
            while not self.startposition_check(task):
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = MPObservation(self.current_env)
        else:
            while self.completion_check():  # Ensure that they are not the same
                self.current_env, _, _, _ = self.env.reset(settle_time=50)
                self.current = MPObservation(self.current_env)

        self.steps_left = self.max_steps
        self.terminal = False
        return self.current, self.goal

    def wrapped_reset(self) -> SceneObservation:  # type: ignore
        """Resets the environment for data collection"""
        self.reset()
        return self.make_tapas_format(self.current_env)

    def step_exp1(
        self,
        task: Skill,
        verbose: bool = False,
        p_empty: float = 0.0,
        p_rand: float = 0.0,
    ) -> tuple[float, bool, MPObservation]:
        sample = random.random()
        if sample < p_empty:  # 0-p_empty>
            logger.warning("Taking Empty Step")
            pass
        elif sample < p_empty + p_rand:  # 0-p_empty + p_rand>
            logger.warning("Taking Random Step")
            self.give_control(random.choice(self.tasks), verbose=verbose)
        else:  # The rest
            self.give_control(task, verbose=verbose)
        self.steps_left -= 1
        reward, done = self.evaluate()
        return reward, done, self.current

    def give_control(
        self,
        skill: Skill,
        predict_at_once: bool = False,
        as_batch: bool = False,
        control_duration: int = -1,
    ):
        skill.prepare(
            predict_at_once=predict_at_once,
            as_batch=as_batch,
            control_duration=control_duration,
        )
        try:
            while (
                action := skill.predict(self.current, self.goal, self.states)
            ) is None:
                self.current_env, _, _, _ = self.env.step(action, self.config.debug_vis)
                self.current = MPObservation(self.current_env)
        except Exception as e:
            # At some point the model crashes.
            # Have to debug if its because of bad input but seems to be not relevant for training
            logger.error(f"Error happened during step: {e}")

    def wrapped_direct_step(
        self, action: np.ndarray, verbose: bool = False
    ) -> SceneObservation:  # type: ignore
        """Directly step the environment with an action."""
        logger.warning(
            "Direct stepping the environment can break internal states and is just for tapas purposes."
        )
        next_calvin, step_reward, _, _ = self.env.step(action, self.config.debug_vis)
        ee_delta = self.env.compute_ee_delta(self.current_env, next_calvin)
        self.current_env.action = torch.Tensor(ee_delta)
        self.current_env.reward = torch.Tensor([step_reward])
        if verbose:
            print(self.current_env.ee_pose)
            print(self.current_env.ee_state)
        tapas_obs = self.make_tapas_format(self.current_env)
        self.current_env = next_calvin
        self.current = MPObservation(self.current_env)
        return tapas_obs

    def close(self):
        self.env.close()

    def evaluate(self) -> tuple[float, bool]:
        if self.terminal:
            raise UserWarning(
                "Episode already ended. Please reset the evaluator with the new goal and state."
            )
        if self.config.reward_mode is RewardMode.SPARSE:
            if self.task:
                if self.endposition_check(self.task):
                    return self.config.max_reward, True
                else:
                    return self.config.min_reward, False
            if self.completion_check():
                return self.config.max_reward, True
            else:
                return self.config.min_reward, False if self.steps_left > 0 else True
        if self.config.reward_mode is RewardMode.ONOFF:
            raise NotImplementedError("Reward Mode not implemented.")
        if self.config.reward_mode is RewardMode.RANGE:
            raise NotImplementedError("Reward Mode not implemented.")

    def startposition_check(self, task: Skill) -> bool:
        ##### Checking if start position is reached
        for state in self.states:
            if state.name in task.task_parameters_keys:
                if task.reversed:
                    value = task.anti_task_parameters[state.name]
                else:
                    value = task.task_parameters[state.name]
                start_reached = state.evaluate(
                    self.current.states[state.name],
                    value,
                    self.eval_surfaces,
                )
            if not start_reached:
                return False  # Early exit if start position is not reached
        return True

    def endposition_check(self, task: Skill) -> bool:
        ##### Checking if end position is reached
        for state in self.states:
            if state.name in task.task_parameters_keys:
                if task.reversed:
                    value = task.anti_task_parameters[state.name]
                else:
                    value = task.anti_task_parameters[state.name]
                end_reached = state.evaluate(
                    self.current.states[state.name],
                    value,
                    self.eval_surfaces,
                )
            if not end_reached:
                return False  # Early exit if end position is not reached

    def completion_check(self) -> bool:
        ##### Checking if goal is reached
        for state in self.states:
            goal_reached = state.evaluate(
                self.current.states[state.name],
                self.goal.states[state.name],
                self.eval_surfaces,
            )
            # print(f"State {state.name} is {goal_reached}")
            if not goal_reached:
                return False  # Early exit if goal is already not reached
        return True
