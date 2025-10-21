from dataclasses import dataclass
from typing import Optional
from src.core.agents.agent import BaseAgent
from src.core.modules.buffer_module import BufferModule
from src.core.modules.reward_module import RewardModule, SparseRewardModule
from src.core.modules.storage_module import StorageModule
from src.core.observation import BaseObservation
from src.core.skill import BaseSkill


class TreeNode:
    def __init__(
        self,
        observation: BaseObservation,
        parent: Optional["TreeNode"] = None,
        children: list[Optional["TreeNode"]] = [],
        depth: int = 0,
        distance_to_skill_start: float = float("inf"),
        distance_to_goal: float = float("inf"),
    ):
        self.observation = observation
        self.parent = parent
        self.children = children
        self.depth = depth
        self.distance_to_goal = distance_to_goal


@dataclass
class SearchTreeAgentConfig:
    distance_threshold: float = 0.1
    max_tree_depth: int = 5
    num_simulated_paths: int = 50
    replan_every_step: bool = True


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
            self.root = TreeNode(observation=obs)
            self._expand_tree(self.root, obs)
            self._find_path(goal)
            self.path_index = 0
            self.current = self.root

        if self.path_index < len(self.path):
            skill = self.storage_module.skills[
                self.path[self.path_index]
            ]  # Next skill in path
            self.path_index += 1
        else:
            raise Exception("No valid path found in search tree.")
        return skill

    def _find_path(self, goal: BaseObservation):
        """Find a path from root to the observation closest to goal in the entire tree"""
        if not self.root:
            raise Exception("Search tree root is not initialized.")

        # Find the node closest to goal in the entire tree
        closest_node = self._find_closest_node_to_goal(self.root, goal)

        # Build path from root to the closest node
        self.path = self._build_path_to_node(closest_node)

    def _find_closest_node_to_goal(
        self, node: TreeNode, goal: BaseObservation
    ) -> TreeNode:
        """Recursively search entire tree to find node closest to goal"""
        closest_node = node
        is_equal = self.eval_module.is_equal(node.observation, goal)

        # Check all children recursively
        for child in node.children:
            if child is not None:  # Skip None children
                child_closest = self._find_closest_node_to_goal(child, goal)
                child_distance = self._compute_distance(child_closest.observation, goal)

                if child_distance < min_distance:
                    min_distance = child_distance
                    closest_node = child_closest

        return closest_node

    def _build_path_to_node(self, node: TreeNode) -> list[int]:
        """Build path from root to target node by traversing parent pointers"""
        path = []
        current = node

        # Build path backwards from target to root
        while current:
            path.append(current)
            current = current.parent

        # Reverse to get path from root to target
        path.reverse()
        return path

    def _expand_tree(self, node: TreeNode, goal: BaseObservation):
        """Expand tree by applying skill postconditions"""
        if node.depth >= self.config.max_tree_depth:
            return

        # Try applying each available skill
        for skill in self.storage_module.skills:
            # Check if skill's preconditions are satisfied
            start_distance = self.eval_module.skill_start_distance(
                skill, node.observation, goal
            )
            distance = start_distance / len(self.storage_module.states)

            if distance < self.config.distance_threshold:
                # Simulate applying the skill by using its postcondition
                simulated_obs = self._apply_skill_postcondition(node.observation, skill)

                child_node = TreeNode(
                    observation=simulated_obs,
                    parent=node,
                    depth=node.depth + 1,
                )
                node.children.append(child_node)

                # Recursively expand (limited by depth)
                if child_node.depth < self.config.max_tree_depth:
                    self._expand_tree(child_node, goal)
            else:
                node.children.append(None)

    def _apply_skill_postcondition(
        self, current_obs: BaseObservation, skill: BaseSkill
    ) -> BaseObservation:
        """Apply skill's postcondition to predict next observation"""
        raise NotImplementedError(
            "_apply_skill_postcondition method not implemented yet."
        )

    def _select_best_skill(self, node: TreeNode, goal: BaseObservation) -> BaseSkill:
        """Select the best skill based on tree search"""
        if not node.children:
            # No children - return a random applicable skill
            applicable_skills = [
                skill
                for skill in self.storage_module.skills
                if skill.precondition_satisfied(node.observation)
            ]
            return (
                applicable_skills[0]
                if applicable_skills
                else self.storage_module.skills[0]
            )

        # Evaluate each child path
        best_skill = None
        best_score = float("-inf")

        for child in node.children:
            score = self._evaluate_path(child, goal)
            if score > best_score:
                best_score = score
                best_skill = child.skill_applied

        return best_skill if best_skill else self.storage_module.skills[0]

    def _evaluate_path(self, node: TreeNode, goal: BaseObservation) -> float:
        """Evaluate how good a path is for reaching the goal"""
        # Simple evaluation: distance to goal + depth penalty
        goal_distance = self._compute_distance(node.observation, goal)
        depth_penalty = node.depth * 0.1  # Prefer shorter paths

        # Higher score is better (closer to goal, shorter path)
        score = -goal_distance - depth_penalty

        # Bonus if we've reached the goal
        if goal_distance < self.config.distance_threshold:
            score += 10.0

        return score

    def _compute_distance(self, obs1: BaseObservation, obs2: BaseObservation) -> float:
        """Compute distance between two observations"""
        # This depends on your observation structure
        # Example implementation:
        try:
            # If observations have a distance method
            if hasattr(obs1, "distance_to"):
                return obs1.distance_to(obs2)

            # Or compute based on states
            distance = 0.0
            for state in self.storage_module.states:
                val1 = state.value(obs1.top_level_observation[state.name])
                val2 = state.value(obs2.top_level_observation[state.name])

                # Simple L2 distance
                if hasattr(val1, "norm") and hasattr(val2, "norm"):
                    distance += (val1 - val2).norm().item()
                else:
                    distance += abs(float(val1) - float(val2))

            return distance

        except Exception:
            return float("inf")  # Fallback

    def feedback(self, reward: float, terminal: bool):
        """Update current node based on actual environment feedback"""
        if self.current and terminal:
            # Reset tree for next episode
            self.root = None
            self.current = None

        # You could also update node values based on actual reward
        # This would require storing values in TreeNode

    def learn(self) -> bool:
        # For search tree, learning might involve updating distance metrics
        # or skill success probabilities
        return False  # Continue training

    def save(self, tag: str = ""):
        # Save tree statistics or learned parameters
        pass

    def load(self):
        # Load saved parameters
        pass
