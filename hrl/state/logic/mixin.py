# Mixin classes for common functionality
from functools import cached_property
from typing import Dict, Optional
import numpy as np
import torch


class QuaternionMixin:
    """Mixin for quaternion normalization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion and ensure positive w component."""
        x = x / torch.linalg.norm(x)
        if x[3] < 0:
            return -x
        return x

    def _quaternion_mean(self, quaternions: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean quaternion using the eigenvector method.
        quaternions: tensor of shape [N, 4] (x, y, z, w)
        Returns: mean quaternion [4] in (x, y, z, w) format
        """
        # Swap to (w, x, y, z) for computation
        quats = quaternions[:, [3, 0, 1, 2]]
        quats = quats / quats.norm(dim=1, keepdim=True)
        A = quats.t() @ quats
        _, eigenvectors = torch.linalg.eigh(A)
        mean_quat = eigenvectors[:, -1]
        # Ensure positive scalar part
        if mean_quat[0] < 0:
            mean_quat = -mean_quat
        # Swap back to (x, y, z, w)
        mean_quat_xyzw = mean_quat[[1, 2, 3, 0]]
        return self._normalize_quat(mean_quat_xyzw)

    def _quaternion_distance(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """Calculate the angular distance between two quaternions."""
        dot_product = torch.abs(torch.dot(q1, q2))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        angle = 2 * torch.acos(dot_product)
        return angle.item()


class BoundedMixin:
    """Mixin for success conditions that need bounds"""

    def __init__(
        self,
        lower_bound: float | np.ndarray,
        upper_bound: float | np.ndarray,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    @property
    def lower_bound(self) -> float | np.ndarray:
        return self._lower_bound

    @property
    def upper_bound(self) -> float | np.ndarray:
        return self._upper_bound

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a value x to the range [0, 1] based on bounds."""
        return (x - self.lower_bound) / (self.upper_bound - self.lower_bound)


class ThresholdMixin:
    """Mixin for success conditions that use thresholds"""

    def __init__(self, threshold: float = 0.05, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold


class RelThresholdMixin(BoundedMixin, ThresholdMixin):
    """Mixin that provides relative threshold calculation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @cached_property
    def relative_threshold(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.threshold * (self.upper_bound - self.lower_bound)


class AreaCheckMixin:
    """Mixin for area-based success conditions"""

    def __init__(self, surfaces: Dict[str, np.ndarray], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surfaces = surfaces

    def check_area(self, x: torch.Tensor) -> Optional[str]:
        """Check if the point x is in any of the defined areas."""
        for name, surface in self.surfaces.items():
            if self.point_in_polygon(x.numpy(), surface):
                return name
        return None

    def _check_area_states(self, obs: torch.Tensor, goal: torch.Tensor) -> bool:
        """Check if both obs and goal are in the same defined area."""
        obs_area = self.check_area(obs)
        goal_area = self.check_area(goal)
        return obs_area is not None and (obs_area == goal_area)

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
