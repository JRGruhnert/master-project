from dataclasses import dataclass
from typing import Optional
from src.core.agents.agent import AgentConfig, BaseAgent
from src.core.modules.buffer_module import BufferModule
from src.core.modules.reward_module import SparseRewardModule
from src.core.modules.storage_module import StorageModule
from src.core.observation import BaseObservation
from src.core.skills.skill import BaseSkill
from loguru import logger


class TreeNode:
    def __init__(
        self,
        observation: BaseObservation,
        parent: Optional["TreeNode"] = None,
        skill_id: Optional[int] = None,
        burnt_skills: Optional[set[int]] = None,
        distance_to_obs: float = 0.0,
        distance_to_goal: float = float("inf"),
        distance_to_skill: float = float("inf"),
    ):
        self.observation = observation  # Need that to calculate following distances
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
    max_depth: int = 6
    allow_skill_reuse: bool = False
    replan_every_step: bool = False


class SearchTreeAgent(BaseAgent):

    def __init__(
        self,
        config: SearchTreeAgentConfig,
        buffer_module: BufferModule,
        storage_module: StorageModule,
        reward_module: SparseRewardModule,
    ):
        self.config: SearchTreeAgentConfig = config
        self.buffer_module: BufferModule = buffer_module
        self.storage_module: StorageModule = storage_module  # Access to skills
        self.eval_module: SparseRewardModule = reward_module

        self.root: Optional[TreeNode] = None
        self.current: Optional[TreeNode] = None
        self.path: list[int] = []

    def act(
        self,
        obs: BaseObservation,
        goal: BaseObservation,
    ) -> BaseSkill:
        # Initialize root if first observation
        if self.config.replan_every_step or self.root is None:
            print(f"Replanning search tree...")
            self.root = TreeNode(observation=obs)
            self._expand_tree(0, self.root, goal)
            self._find_path(goal)
            self.path_index = 0
            self.current = self.root

        print(f"Current path: {self.path}")
        if self.path_index < len(self.path):
            skill = self.storage_module.skills[
                self.path[self.path_index]
            ]  # Next skill in path
            self.path_index += 1
        else:
            raise Exception("No valid path found in search tree.")
        return skill

    def _expand_tree(self, depth: int, node: TreeNode, goal: BaseObservation):
        """Expand tree by applying skill postconditions"""
        # print(f"Expanding tree at depth {depth}...")
        if depth >= self.config.max_depth:
            # print(f"Max depth reached, stopping expansion.")
            return

        # Try applying each available skill
        for skill in self.storage_module.skills:
            # Check if skill's preconditions are satisfied
            if self.config.allow_skill_reuse or skill.id not in node.burnt_skills:
                skill_distance = self.eval_module.distance_to_skill(
                    node.observation, goal, skill
                )
                if skill_distance < self.config.distance_threshold:
                    # Simulate applying the skill by using its postcondition
                    simulated_obs = self._apply_skill_postcondition(
                        node.observation, skill
                    )
                    goal_distance = self.eval_module.distance_to_goal(
                        simulated_obs, goal
                    )
                    child_node = TreeNode(
                        observation=simulated_obs,
                        parent=node,
                        skill_id=skill.id,
                        burnt_skills=node.burnt_skills.union({skill.id}),
                        distance_to_obs=node.distance_to_obs + skill_distance,
                        distance_to_goal=goal_distance,
                        distance_to_skill=skill_distance,
                    )
                    node.children.update({skill.id: child_node})

                    # Recursively expand (limited by depth)
                    self._expand_tree(depth + 1, child_node, goal)

    def _find_path(self, goal: BaseObservation):
        """Find a path from root to the observation closest to goal in the entire tree"""
        if not self.root:
            raise Exception("Search tree root is not initialized.")
        print(f"Finding path to goal...")
        best_node = self._find_best_node(self.root, goal)
        self.path = self._build_path_to_node(best_node)

    def _find_best_node(self, node: TreeNode, goal: BaseObservation) -> TreeNode:
        """Recursively search entire tree to find node closest to goal"""
        best_node = node
        # Check all children recursively
        for skill_id, child in node.children.items():
            child_closest = self._find_best_node(child, goal)

            if (
                child_closest.distance_to_goal < best_node.distance_to_goal
                and child_closest.distance_to_obs < best_node.distance_to_obs
            ):
                best_node = child_closest
            elif child_closest.distance_to_goal < best_node.distance_to_goal:
                logger.debug(
                    f"Tie-breaker: prefer node with lower distance to goal {child_closest.distance_to_goal} < {best_node.distance_to_goal}"
                )
                best_node = child_closest
            elif child_closest.distance_to_obs < best_node.distance_to_obs:
                logger.debug(
                    f"Tie-breaker: prefer node with lower distance to obs {child_closest.distance_to_obs} < {best_node.distance_to_obs}"
                )
                best_node = child_closest

        return best_node

    def _build_path_to_node(self, node: TreeNode) -> list[int]:
        """Build path from root to target node by traversing parent pointers"""
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
        return False  # Continue training

    def save(self, tag: str = ""):
        # No parameters to save
        pass

    def load(self):
        # No parameters to load
        pass
