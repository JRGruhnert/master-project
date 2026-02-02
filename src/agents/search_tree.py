from dataclasses import dataclass
from typing import Optional

import torch
from src.agents.agent import AgentConfig, Agent
from src.modules.buffer import Buffer
from src.modules.evaluators.tree import TreeEvaluator, TreeEvaluatorConfig
from src.modules.storage import Storage
from src.observation.observation import StateValueDict
from src.skills.skill import Skill
from src.skills.empty import EmptySkill
from loguru import logger
from heapq import heapify, heappop, heappush


class TreeNode:
    def __init__(
        self,
        obs: StateValueDict,
        path: Optional[list[int]] = None,
        distance_to_obs: float = 0.0,
        distance_to_goal: float = float("inf"),
        distance_to_skill: float = float("inf"),
    ):
        self.obs = obs  # Need that to calculate following distances
        self.path = path if path is not None else []
        self.distance_to_obs = distance_to_obs
        self.distance_to_goal = distance_to_goal
        self.distance_to_skill = distance_to_skill

    def __lt__(self, other: "TreeNode") -> bool:
        # Compare based on distance to goal, then distance to obs, then distance to skill
        return (
            self.distance_to_goal,
            self.distance_to_skill,
            self.distance_to_obs,
            self.depth,
        ) < (
            other.distance_to_goal,
            other.distance_to_skill,
            other.distance_to_obs,
            other.depth,
        )

    @property
    def depth(self) -> int:
        return len(self.path)


@dataclass
class SearchTreeAgentConfig(AgentConfig):
    distance_threshold: float = 0.2
    max_depth: int = 15
    allow_skill_reuse: bool = False
    replan_every_step: bool = False
    max_epochs: int = 5
    beam_size: int = 5
    max_nodes: int = 10000  # guard to avoid infinite loops
    evaluator_config: TreeEvaluatorConfig = TreeEvaluatorConfig()


class SearchTreeAgent(Agent):

    def __init__(
        self,
        config: SearchTreeAgentConfig,
        storage: Storage,
        buffer: Buffer,
    ):
        self.config = config
        self.storage = storage
        self.buffer = buffer
        self.evaluator = TreeEvaluator(self.config.evaluator_config, self.storage)

        self.heap: list[TreeNode] = []
        self.solution = None
        self.index: int = 0
        self.current_epoch = 0

    def act(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
    ) -> Skill:
        # Initialize root if first observation
        if self.config.replan_every_step or not self.heap:
            self.goal = goal
            self.index = 0
            self.heap: list[TreeNode] = [TreeNode(obs=obs)]
            heapify(self.heap)
            self.solution = self.best_first_search()

        if not self.solution or self.index >= len(self.solution.path):
            # print("Empty Skill taken cause of index too high.")
            skill = EmptySkill()
        else:
            skill = self.storage.skills[
                self.solution.path[self.index]
            ]  # Next skill in path
            assert (
                skill.id == self.solution.path[self.index]
            )  # Just for checking if its really the same
        self.index += 1
        self.buffer.act_values_tree(obs, goal, skill.id)
        return skill

    def best_first_search(self) -> Optional[TreeNode]:
        """Expand tree by applying skill postconditions"""
        nodes_expanded = 0

        while self.heap and nodes_expanded < self.config.max_nodes:
            current = heappop(self.heap)
            nodes_expanded += 1

            _, done = self.evaluator.step(current.obs, self.goal)
            if done:
                logger.debug(f"Goal reached. Stopping search.")
                return current

            # Try applying each available skill
            for skill in self.storage.skills:
                if self.config.allow_skill_reuse or skill.id not in current.path:
                    skill_distance = self.evaluator.distance_to_skill(
                        current.obs,
                        skill,
                    )
                    if skill_distance < self.config.distance_threshold:
                        # Simulate applying the skill by using its postcondition
                        simulated_obs = self._apply_skill_postcondition(
                            current.obs,
                            skill,
                        )
                        goal_distance = self.evaluator.distance_to_goal(
                            simulated_obs,
                            self.goal,
                        )

                        if self.config.max_depth == -1 or (
                            current.depth + 1 < self.config.max_depth
                        ):  # Only push if max depth not reached
                            heappush(
                                self.heap,
                                TreeNode(
                                    obs=simulated_obs,
                                    path=current.path + [skill.id],
                                    distance_to_obs=current.distance_to_obs
                                    + skill_distance,
                                    distance_to_goal=goal_distance,
                                    distance_to_skill=skill_distance,
                                ),
                            )
        return None  # No solution found within limits

    def _apply_skill_postcondition(
        self, current: StateValueDict, skill: Skill
    ) -> StateValueDict:
        """Apply skill's postcondition to predict next observation"""

        next_obs = current.clone()
        # Apply skill postconditions
        for state_name, target_value in skill.postcons.items():
            # NOTE: HACKY for the Flip State (But this is just a baseline so not worth it to generalize)
            if state_name == "base__button_scalar":
                if skill.name == "PressButton":
                    next_obs[state_name] = (
                        torch.Tensor([1.0])
                        if next_obs[state_name] < 1.0
                        else torch.Tensor([0.0])
                    )
            else:
                next_obs[state_name] = target_value

        return next_obs

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        """Update current node based on actual environment feedback"""
        if terminal:
            # Reset tree for next episode
            self.heap = []  # Clear tree
        return self.buffer.feedback(reward, success, terminal)

    def learn(self) -> bool:
        # No Model to train in search tree agent
        self.buffer.save(self.storage.buffer_saving_path, self.current_epoch)
        # Update Epoch
        self.current_epoch += 1
        # Clear buffer
        self.buffer.clear()

        self.heap = []  # Clear tree
        return self.config.max_epochs == self.current_epoch - 1

    def save(self, tag: str = ""):
        # No parameters to save
        pass

    def load(self):
        # No parameters to load
        pass

    def metadata(self) -> dict:
        return {
            "distance_threshold": self.config.distance_threshold,
            "max_depth": self.config.max_depth,
            "allow_skill_reuse": self.config.allow_skill_reuse,
            "replan_every_step": self.config.replan_every_step,
            "max_epochs": self.config.max_epochs,
            "beam_size": self.config.beam_size,
            "max_nodes": self.config.max_nodes,
        }

    def metrics(self) -> dict:
        return {}
