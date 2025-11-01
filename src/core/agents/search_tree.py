from dataclasses import dataclass
from typing import Optional
from src.core.agents.agent import AgentConfig, BaseAgent
from src.core.modules.buffer_module import BufferModule
from src.core.modules.reward_module import RewardModule
from src.core.modules.storage_module import StorageModule
from src.core.observation import BaseObservation
from src.core.skills.skill import BaseSkill, EmptySkill
from loguru import logger


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


@dataclass
class SearchTreeAgentConfig(AgentConfig):
    distance_threshold: float = 0.2
    max_depth: int = -1
    allow_skill_reuse: bool = False
    replan_every_step: bool = False
    max_epochs: int = 5
    beam_size: int = 5


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

        self.node: Optional[TreeNode] = None
        self.index: int = 0
        self.current_epoch = 0

    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
    ) -> BaseSkill:
        # Initialize root if first observation
        if (
            self.config.replan_every_step
            or not self.node
            or self.index == len(self.node.path)
        ):
            self.index = 0
            self.node = TreeNode(obs=obs)
            candidate = self.expand_tree(0, self.node, goal)
            # print(f"Candidate: {candidate}")
            if candidate:
                self.node = candidate
                print(f"Path: {candidate.path}")
            # If no path found, choose random skill
            if len(self.node.path) == 0:
                print("No path found in search tree.")
                logger.warning("No path found in search tree.")

        if self.index >= len(self.node.path):
            # print("Empty Skill taken cause of index too high.")
            skill = EmptySkill()
        else:
            skill = self.storage_module.skills[
                self.node.path[self.index]
            ]  # Next skill in path
            assert (
                skill.id == self.node.path[self.index]
            )  # Just for checking if its really the same
        self.index += 1
        self.buffer_module.act_values_tree(obs, goal, skill.id)
        return skill

    def expand_tree(
        self, depth: int, origin: TreeNode, goal: BaseObservation
    ) -> TreeNode | None:
        """Expand tree by applying skill postconditions"""
        if depth + 1 >= self.config.max_depth:
            logger.debug(f"Max depth reached, stopping expansion.")
            return None

        _, done = self.reward_module.step(origin.obs, goal)
        if done:
            logger.debug(f"Goal reached. Stopping search.")
            return origin

        candidates = []
        # Try applying each available skill
        for skill in self.storage_module.skills:
            # Check if skill's preconditions are satisfied
            if self.config.allow_skill_reuse or skill.id not in origin.path:
                skill_distance = self.reward_module.distance_to_skill(
                    origin.obs,
                    goal,
                    skill,
                )
                if skill_distance < self.config.distance_threshold:
                    # Simulate applying the skill by using its postcondition
                    simulated_obs = self._apply_skill_postcondition(
                        origin.obs,
                        skill,
                    )
                    goal_distance = self.reward_module.distance_to_goal(
                        simulated_obs,
                        goal,
                    )
                    candidates.append(
                        TreeNode(
                            obs=simulated_obs,
                            path=origin.path + [skill.id],
                            distance_to_obs=origin.distance_to_obs + skill_distance,
                            distance_to_goal=goal_distance,
                            distance_to_skill=skill_distance,
                        )
                    )

        for candidate in candidates:
            branch_candidate = self.expand_tree(depth + 1, candidate, goal)

            if branch_candidate:  # Stops searching the tree if a candidate was found
                return branch_candidate

        logger.debug(f"Current depth is: {depth}")
        return None  # No candidate in that branch

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
            self.node = None
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

        self.node = None
        return self.config.max_epochs == self.current_epoch - 1

    def save(self, tag: str = ""):
        # No parameters to save
        pass

    def load(self):
        # No parameters to load
        pass
