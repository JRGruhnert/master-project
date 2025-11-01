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
        parent: Optional["TreeNode"] = None,
        skill_id: Optional[int] = None,
        burnt_skills: Optional[set[int]] = None,
        distance_to_obs: float = 0.0,
        distance_to_goal: float = float("inf"),
        distance_to_skill: float = float("inf"),
    ):
        self.obs = obs  # Need that to calculate following distances
        self.parent = parent
        self.skill_id = skill_id  # Skill that led to this node
        self.burnt_skills = burnt_skills if burnt_skills is not None else set()
        self.children: dict[int, "TreeNode"] = {}  # skill_id -> TreeNode
        self.distance_to_obs = distance_to_obs
        self.distance_to_goal = distance_to_goal
        self.distance_to_skill = distance_to_skill


@dataclass
class SearchTreeAgentConfig(AgentConfig):
    distance_threshold: float = 0.1
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

        self.root: Optional[TreeNode] = None
        self.current: Optional[TreeNode] = None
        self.path: list[int] = []
        self.path_index: int = 0
        self.current_epoch = 0

    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
    ) -> BaseSkill:
        # Initialize root if first observation
        if (
            self.config.replan_every_step
            or self.root is None
            or self.path_index == len(self.path)
        ):
            # print(f"Replanning search tree...")
            self.root = TreeNode(obs=obs)
            self._expand_tree(0, self.root, goal)
            self._find_path(goal)
            self.path_index = 0
            self.current = self.root

            # If no path found, choose random skill
            if len(self.path) == 0:
                print("No path found in search tree.")
                # logger.warning("No path found in search tree.")

        if self.path_index == len(self.path):
            print("Empty Skill taken cause of index too high.")
            skill = EmptySkill()
        else:
            # print(f"Current path: {self.path}")
            # print(f"Path index: {self.path_index}")s
            skill = self.storage_module.skills[
                self.path[self.path_index]
            ]  # Next skill in path
        self.path_index += 1
        self.buffer_module.act_values_tree(obs, goal, skill.id)
        return skill

    def _expand_tree(self, depth: int, node: TreeNode, goal: BaseObservation):
        """Expand tree by applying skill postconditions"""
        # print(f"Expanding tree at depth {depth}...")
        if depth >= self.config.max_depth:
            logger.debug(f"Max depth reached, stopping expansion.")
            return

        _, done = self.reward_module.step(node.obs, goal)
        if done:  # Very close to goal
            logger.debug(f"Goal reached. Stopping expansion in this branch.")
            return

        candidate_children = []
        # Try applying each available skill
        for skill in self.storage_module.skills:
            # Check if skill's preconditions are satisfied
            if self.config.allow_skill_reuse or skill.id not in node.burnt_skills:
                skill_distance = self.reward_module.distance_to_skill(
                    node.obs,
                    goal,
                    skill,
                )
                if skill_distance < self.config.distance_threshold:
                    # Simulate applying the skill by using its postcondition
                    simulated_obs = self._apply_skill_postcondition(
                        node.obs,
                        skill,
                    )
                    goal_distance = self.reward_module.distance_to_goal(
                        simulated_obs,
                        goal,
                    )
                    child_node = TreeNode(
                        obs=simulated_obs,
                        parent=node,
                        skill_id=skill.id,
                        burnt_skills=node.burnt_skills.union({skill.id}),
                        distance_to_obs=node.distance_to_obs + skill_distance,
                        distance_to_goal=goal_distance,
                        distance_to_skill=skill_distance,
                    )
                    # Add child node to candidates
                    candidate_children.append((skill.id, child_node, goal_distance))
        # Beam search: Sort candidates by distance to goal and keep top N
        candidate_children.sort(key=lambda x: x[2])  # Sort by distance to goal
        # best_candidates = candidate_children[: self.config.beam_size]
        best_candidates = candidate_children
        for skill_id, child_node, _ in best_candidates:
            node.children.update({skill_id: child_node})
            # Recursively expand (limited by depth)
            self._expand_tree(depth + 1, child_node, goal)
        logger.debug(f"Current depth is: {depth}")

    def _find_path(self, goal: BaseObservation):
        """Find a path from root to the observation closest to goal in the entire tree"""
        logger.debug("Searching a path in tree.")
        if not self.root:
            raise Exception("Search tree root is not initialized.")
        # print(f"Finding path to goal...")
        best_node = self._find_best_node(self.root, goal)
        self.path = self._build_path_to_node(best_node)

    def _find_best_node(self, node: TreeNode, goal: BaseObservation) -> TreeNode:
        """Recursively search entire tree to find node closest to goal"""
        best_node = node
        # Check all children recursively
        for _, child in node.children.items():
            child_closest = self._find_best_node(child, goal)

            if child_closest.distance_to_goal < best_node.distance_to_goal:
                best_node = child_closest

        return best_node

    def _build_path_to_node(self, node: TreeNode) -> list[int]:
        """Build path from root to target node by traversing parent pointers"""
        logger.debug("Building the path.")
        path = []
        current = node

        # Build path backwards from target to root
        while current and current.skill_id is not None:
            path.append(current.skill_id)
            current = current.parent

        # Reverse to get path from root to target
        path.reverse()
        return path

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
        if self.current and terminal:
            # Reset tree for next episode
            self.root = None
            self.current = None
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

        self.root = None
        self.current = None
        return self.config.max_epochs == self.current_epoch - 1

    def save(self, tag: str = ""):
        # No parameters to save
        pass

    def load(self):
        # No parameters to load
        pass
