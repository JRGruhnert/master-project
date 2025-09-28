from abc import ABC, abstractmethod
import torch
import numpy as np

from abc import ABC, abstractmethod
from functools import cached_property
import torch
import numpy as np


class SkillCondition(ABC):
    """Abstract base class for skill evaluation strategies."""

    @abstractmethod
    def evaluate(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        reset: bool = False,
    ) -> bool:
        """Evaluate goal condition for the given state."""
        raise NotImplementedError("Subclasses must implement the evaluate method.")

    @property
    def threshold(self) -> float:
        """Returns the threshold for the state."""
        return 0.05

    @cached_property
    def relative_threshold(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.threshold * (self.upper_bound - self.lower_bound)


class DefaultSkillCondition(SkillCondition):
    """Default skill condition based on Euclidean distance."""

    def evaluate(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        reset: bool = False,
    ) -> bool:
        distance = torch.norm(obs - goal).item()
        return distance < self.relative_threshold


class AreaSkillCondition(SkillCondition):
    """Skill condition based on area matching."""

    def __init__(self, surfaces: dict[str, np.ndarray]):
        self.surfaces = surfaces

    def evaluate(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        return self.check_area_states(obs, goal)

    def area_tapas_override(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override the area check for TAPAS.
        """
        area = self.check_area(x)
        if area == "closed_drawer":
            x[1] -= 0.17  # Drawer Offset
        return x  # Return original point if no area match

    def check_area(self, x: torch.Tensor) -> str | None:
        """
        Check if the point x is in any of the defined areas.
        Returns the name of the area or None if not found.
        """
        for name, surface in self.surfaces.items():
            if self.point_in_polygon(x.numpy(), surface):
                return name
        return None

    def check_area_states(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """
        Check if both obs and goal are in the same area.
        """
        obs_area = self.check_area(obs)
        goal_area = self.check_area(goal)
        return obs_area is not None and obs_area == goal_area

    def make_eval_surfaces(
        self, surfaces: dict[str, np.ndarray], padding_percent: float
    ):
        eval_surfaces: dict[str, np.ndarray] = surfaces
        eval_surfaces["table"] = self.add_surface_padding(
            eval_surfaces["table"], padding_percent
        )
        eval_surfaces["drawer_open"][0][0] -= 0.02
        eval_surfaces["drawer_open"][1][0] += 0.02
        eval_surfaces["drawer_closed"][0][0] -= 0.02
        eval_surfaces["drawer_closed"][1][0] += 0.02
        eval_surfaces["drawer_open"][0][1] -= 0.02
        eval_surfaces["drawer_open"][1][1] += 0.02
        eval_surfaces["drawer_closed"][0][1] -= 0.02
        eval_surfaces["drawer_closed"][1][1] += 0.02

        return {k: torch.from_numpy(np.array(v)) for k, v in eval_surfaces.items()}

    def add_surface_padding(self, surface, padding_percent: float):
        """Add padding to surface bounds in x and y directions"""
        # surface = np.array(surface)

        # Get bounds
        x_min, y_min, z_min = surface[0]
        x_max, y_max, z_max = surface[1]

        # Calculate padding amounts
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_padding = x_range * padding_percent / 2  # Divide by 2 for each side
        y_padding = y_range * padding_percent / 2

        # Apply padding (keep z unchanged)
        padded_surface = [
            [x_min - x_padding, y_min - y_padding, z_min],
            [x_max + x_padding, y_max + y_padding, z_max],
        ]

        return padded_surface
