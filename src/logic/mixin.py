# Mixin classes for common functionality
from functools import cached_property
from typing import Dict, Optional
import numpy as np
import torch


class QuaternionMixin:
    """Mixin for quaternion normalization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def normalize_quat(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize quaternion and ensure positive w component."""
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        w = x[..., 3]
        mask = w < 0
        x[mask] = -x[mask]
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
        return self.normalize_quat(mean_quat_xyzw)

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
        lower_bound: list[float],
        upper_bound: list[float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lower_bound = torch.tensor(lower_bound, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, dtype=torch.float32)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
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

    @cached_property
    def relative_threshold(self) -> torch.Tensor:
        """Returns the relative threshold for the state."""
        return self.threshold * (self.upper_bound - self.lower_bound)


class AreaMixin:
    """Mixin for area-based success conditions"""

    def __init__(
        self,
        spawn_surfaces: Dict[str, list[list[float]]],
        eval_surfaces: Dict[str, list[list[float]]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.table: str = "table"
        self.drawer_open: str = "drawer_open"
        self.drawer_closed: str = "drawer_closed"
        self.drawer: str = "drawer"
        self.spawn_surfaces = self.make_surfaces(spawn_surfaces)
        self.eval_surfaces = self.make_surfaces(eval_surfaces)

    def check_eval_area(self, x: torch.Tensor) -> Optional[str]:
        """Check if the point x is in any of the defined areas."""
        for name, (min_corner, max_corner) in self.eval_surfaces.items():
            # min_corner = min_corner.to(x.device, dtype=x.dtype)
            # max_corner = max_corner.to(x.device, dtype=x.dtype)
            # print("Checking area:", name)
            # print("Min corner:", min_corner)
            # print("Max corner:", max_corner)
            # print("Point to check:", x)
            # print(
            #    "Check result:", torch.all(x >= min_corner), torch.all(x <= max_corner)
            # )
            # ✅ Let's manually check each element
            # for i in range(len(x)):
            #    print(
            #        f"  Element {i}: {x[i]} >= {min_corner[i]} = {x[i] >= min_corner[i]}"
            #    )
            #    print(
            #        f"  Element {i}: {x[i]} <= {max_corner[i]} = {x[i] <= max_corner[i]}"
            #    )

            if torch.all(x >= min_corner) and torch.all(x <= max_corner):
                return name
            # else:
            #    print(torch.all(x >= min_corner), torch.all(x <= max_corner))
        return None

    def check_spawn_area(self, x: torch.Tensor) -> Optional[str]:
        """Check if the point x is in any of the defined spawn areas."""
        for name, (min_corner, max_corner) in self.spawn_surfaces.items():
            if torch.all(x >= min_corner) and torch.all(x <= max_corner):
                return name
        return None

    def check_area_type_discrepancy(self, value: torch.Tensor) -> bool:
        """Check if the given value is in the same eval and spawn area."""
        eval_area = self.check_eval_area(value)
        spawn_area = self.check_spawn_area(value)
        return eval_area == spawn_area

    def check_area_similarity(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Check if both obs and goal are in the same defined area."""
        current_area = self.check_eval_area(current)
        # print("Current area:", current_area)
        goal_area = self.check_eval_area(goal)
        # print("Goal area:", goal_area)
        # print("Area match:", current_area == goal_area)
        return current_area == goal_area

    def area_tapas_override(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override the area check for TAPAS.
        """
        area = self.check_spawn_area(x)
        # print("Override area:", area)
        y = x.clone()  # ✅ Prevents modifying the original tensor
        if area == self.drawer_closed:
            y[1] -= 0.17  # Drawer Offset
        return y  # Return original point if no area match

    def make_surfaces(
        self,
        surfaces: dict[str, list[list[float]]],
    ):
        return {k: torch.from_numpy(np.array(v)) for k, v in surfaces.items()}

    def add_surface_padding(
        self,
        surface: list[list[float]],
        padding_percent: float,
    ):
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


class AreaMixin2:
    """Mixin for area-based success conditions"""

    def __init__(self, surfaces: Dict[str, list[list[float]]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table: str = "table"
        self.drawer_open: str = "drawer_open"
        self.drawer_closed: str = "drawer_closed"
        self.drawer: str = "drawer"
        self.eval_surfaces = self.make_eval_surfaces(surfaces)

    def check_area(self, x: torch.Tensor) -> Optional[str]:
        """Check if the point x is in any of the defined areas."""
        for name, (min_corner, max_corner) in self.eval_surfaces.items():
            # min_corner = min_corner.to(x.device, dtype=x.dtype)
            # max_corner = max_corner.to(x.device, dtype=x.dtype)
            # print("Checking area:", name)
            # print("Min corner:", min_corner)
            # print("Max corner:", max_corner)
            # print("Point to check:", x)
            # print(
            #    "Check result:", torch.all(x >= min_corner), torch.all(x <= max_corner)
            # )
            # ✅ Let's manually check each element
            # for i in range(len(x)):
            #    print(
            #        f"  Element {i}: {x[i]} >= {min_corner[i]} = {x[i] >= min_corner[i]}"
            #    )
            #    print(
            #        f"  Element {i}: {x[i]} <= {max_corner[i]} = {x[i] <= max_corner[i]}"
            #    )

            if torch.all(x >= min_corner) and torch.all(x <= max_corner):
                return name
            # else:
            #    print(torch.all(x >= min_corner), torch.all(x <= max_corner))
        return None

    def check_area_similarity(self, current: torch.Tensor, goal: torch.Tensor) -> bool:
        """Check if both obs and goal are in the same defined area."""
        current_area = self.check_area(current)
        # print("Current area:", current_area)
        goal_area = self.check_area(goal)
        # print("Goal area:", goal_area)
        # print("Area match:", current_area == goal_area)
        return current_area == goal_area

    def area_tapas_override(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override the area check for TAPAS.
        """
        area = self.check_area(x)
        # print("Override area:", area)
        y = x.clone()  # ✅ Prevents modifying the original tensor
        if area == self.drawer_closed:
            y[1] -= 0.17  # Drawer Offset
        return y  # Return original point if no area match

    def make_eval_surfaces(
        self,
        surfaces: dict[str, list[list[float]]],
        padding_percent: float = 0.1,
    ):
        eval_surfaces = surfaces.copy()
        eval_surfaces[self.table] = self.add_surface_padding(
            eval_surfaces[self.table], padding_percent
        )
        eval_surfaces[self.drawer_open][0][0] -= 0.02
        eval_surfaces[self.drawer_open][1][0] += 0.02
        eval_surfaces[self.drawer_closed][0][0] -= 0.02
        eval_surfaces[self.drawer_closed][1][0] += 0.02
        eval_surfaces[self.drawer_open][0][1] -= 0.02
        eval_surfaces[self.drawer_open][1][1] += 0.02
        eval_surfaces[self.drawer_closed][0][1] -= 0.02
        eval_surfaces[self.drawer_closed][1][1] += 0.02
        eval_surfaces[self.drawer_open][0][2] -= 0.02
        eval_surfaces[self.drawer_open][1][2] += 0.02
        eval_surfaces[self.drawer_closed][0][2] -= 0.02
        eval_surfaces[self.drawer_closed][1][2] += 0.02
        eval_surfaces[self.table][0][2] -= 0.02
        eval_surfaces[self.table][1][2] += 0.02
        eval_surfaces[self.table][0][1] -= 0.02
        eval_surfaces[self.table][1][1] += 0.02

        return {k: torch.from_numpy(np.array(v)) for k, v in eval_surfaces.items()}

    def add_surface_padding(
        self,
        surface: list[list[float]],
        padding_percent: float,
    ):
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
