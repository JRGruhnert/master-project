from dataclasses import dataclass
from typing import ClassVar, Any

import numpy as np

from heca.data.data import DCScene
from heca.data.entity import Entity


def _clean(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


class FreeEntity(Entity):
    BLOCKS: ClassVar[tuple[str, ...]] = ("state", "pos", "rot")
    @dataclass(kw_only=True)
    class Config(Entity.Config):
        type_id: int = 0
        canonicalize_rot: bool = True

    @property
    def measurement(self) -> dict:
        return {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 3 + self.rot_dim,
                "reg_covar": Entity.REG_COVAR,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        return np.zeros(0)

    def env_state_value(
        self, label: str, x: DCScene, unnormalize_pos=None
    ) -> dict[str, Any]:
        dc = x.get(label)
        pos = unnormalize_pos(dc.pos) if unnormalize_pos is not None else dc.pos
        return {
            f"heca_{label}_pos": pos,
            f"heca_{label}_rot": dc.rot,
            f"heca_{label}_ste": dc.ste,
        }

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        start_val = _clean(obs[f"heca_{label}_loc"][start])
        target_val = _clean(obs[f"heca_{label}_loc"][end])
        return f"{label}_{start_val}_{target_val}"
