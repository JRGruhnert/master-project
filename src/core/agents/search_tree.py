from dataclasses import dataclass
from typing import Optional
from src.core.agents.agent import AgentConfig, BaseAgent
from src.core.modules.buffer_module import BufferModule
from src.core.modules.reward_module import RewardModule
from src.core.modules.storage_module import StorageModule
from src.core.observation import BaseObservation
from src.core.skills.skill import BaseSkill, EmptySkill
from loguru import logger
from heapq import heapify, heappop, heappush


class TreeNode:
    def __init__(
        self,
        obs: BaseObservation,
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
    max_depth: int = -1
    allow_skill_reuse: bool = False
    replan_every_step: bool = False
    max_epochs: int = 5
    beam_size: int = 5
    max_nodes: int = 10000


class SearchTreeAgent(BaseAgent):
    def __init__(
        self,
        config: SearchTreeAgentConfig,
        buffer_module: BufferModule,
        storage_module: StorageModule,
        reward_module: RewardModule,
        max_depth: int,
    ):
        self.config: SearchTreeAgentConfig = config
        self.config.max_depth = max_depth
        self.buffer_module: BufferModule = buffer_module
        self.storage_module: StorageModule = storage_module  # Access to skills
        self.reward_module: RewardModule = reward_module

        self.heap: list[TreeNode] = []
        self.index: int = 0
        self.current_epoch = 0

    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
    ) -> BaseSkill:
        # Initialize root if first observation
        if self.config.replan_every_step or not self.heap:
            self.goal = goal
            self.index = 0
            self.heap: list[TreeNode] = [TreeNode(obs=obs)]
            heapify(self.heap)
            solution = self.best_first_search()

        if not solution or self.index >= len(solution.path):
            # print("Empty Skill taken cause of index too high.")
            skill = EmptySkill()
        else:
            skill = self.storage_module.skills[
                solution.path[self.index]
            ]  # Next skill in path
            assert (
                skill.id == solution.path[self.index]
            )  # Just for checking if its really the same
        self.index += 1
        self.buffer_module.act_values_tree(obs, goal, skill.id)
        return skill

    def best_first_search(self) -> Optional[TreeNode]:
        """Expand tree by applying skill postconditions"""
        nodes_expanded = 0

        while self.heap and nodes_expanded < self.config.max_nodes:
            current = heappop(self.heap)
            nodes_expanded += 1

            _, done = self.reward_module.step(current.obs, self.goal)
            if done:
                logger.debug(f"Goal reached. Stopping search.")
                return current

            # Try applying each available skill
            for skill in self.storage_module.skills:
                if self.config.allow_skill_reuse or skill.id not in current.path:
                    skill_distance = self.reward_module.distance_to_skill(
                        current.obs,
                        self.goal,
                        skill,
                    )
                    if skill_distance < self.config.distance_threshold:
                        # Simulate applying the skill by using its postcondition
                        simulated_obs = self._apply_skill_postcondition(
                            current.obs,
                            skill,
                        )
                        goal_distance = self.reward_module.distance_to_goal(
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
        self, current: BaseObservation, skill: BaseSkill
    ) -> BaseObservation:
        """Apply skill's postcondition to predict next observation"""

        next_obs = current.clone()
        # Apply skill postconditions
        for state_name, target_value in skill.postcons.items():
            # NOTE: HACKY for the Flip State (But this is just a baseline so not worth it to generalize)
            if skill._name in ["PressButton", "PressButtonBack"]:
                # Special handling for button press skills
                if state_name == "button_state":
                    if target_value == 1:
                        next_obs[state_name] = 0  # Toggle button state
                    else:
                        next_obs[state_name] = 1
            else:
                next_obs[state_name] = target_value

        return next_obs

    def feedback(self, reward: float, terminal: bool) -> bool:
        """Update current node based on actual environment feedback"""
        if terminal:
            # Reset tree for next episode
            self.heap = []  # Clear tree
        return self.buffer_module.feedback(reward, terminal)

    def learn(self) -> bool:
        # No Model to train in search tree agent
        self.buffer_module.save(
            self.storage_module.buffer_saving_path, self.current_epoch
        )
        # Update Epoch
        self.current_epoch += 1
        # Clear buffer
        self.buffer_module.clear()

        self.heap = []  # Clear tree
        return self.config.max_epochs == self.current_epoch - 1

    def save(self, tag: str = ""):
        # No parameters to save
        pass

    def load(self):
        # No parameters to load
        pass
