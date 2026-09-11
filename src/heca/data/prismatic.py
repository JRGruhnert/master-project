from dataclasses import dataclass
from typing import ClassVar, Any

import numpy as np

from heca.data.data import DCScene
from heca.data.entity import Entity


class PrismaticEntity(Entity):
    BLOCKS: ClassVar[tuple[str, ...]] = ("state", "pos", "rot", "extra")

    @dataclass(kw_only=True)
    class Config(Entity.Config):
        type_id: int = 2

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 3 + self.rot_dim + 1,
                "reg_covar": Entity.REG_COVAR,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        min_pos = obs[f"heca_{label}_sca_min"]
        max_pos = obs[f"heca_{label}_sca_max"]
        current_pos = obs[f"heca_{label}_sca"]
        relative = (current_pos - min_pos) / (max_pos - min_pos)
        relative = 2 * relative - 1
        return np.array([relative])

    @property
    def extra_sigma(self) -> np.ndarray:
        return np.full(1, self.cfg.sca_sigma)

    def env_state_value(
        self, label: str, x: DCScene, unnormalize_pos=None
    ) -> dict[str, Any]:
        dc = x.get(label)
        pos = unnormalize_pos(dc.pos) if unnormalize_pos is not None else dc.pos
        return {
            f"heca_{label}_pos": pos,
            f"heca_{label}_rot": dc.rot,
            f"heca_{label}_ste": dc.ste,
            f"heca_{label}_sca": float(dc.ext[0]),
        }

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = obs[f"heca_{label}_sca"][start][0]
        target_val = obs[f"heca_{label}_sca"][end][0]
        direction = "a_b" if target_val > start_val else "b_a"
        return f"{label}_{direction}"
