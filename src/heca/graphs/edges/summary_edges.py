from enum import Enum

import numpy as np

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import EntityNode, OptionNode, ValueNode
from heca.data.entity import Entity


class ResidualMode(Enum):
    CONSTANT = "constant"
    POST = "post"


class SummaryEdges(EdgeSet[EntityNode, OptionNode]):

    @property
    def type(self) -> tuple[str, str, str]:
        return ("entity", "summary", "option")

    @property
    def residual_mode(self) -> ResidualMode:
        return ResidualMode.CONSTANT

    def update_attr(self, src: EntityNode, dst: OptionNode, index: int):
        goal_value = dst.data[src.entity].value
        if self.residual_mode == ResidualMode.CONSTANT:
            sigma_pos, sigma_rot = None, None
        else:
            sigma_pos, sigma_rot = self._summary_ruler(src, goal_value)
        self.attrs[index] = self.residual(
            src.data.feature,
            dst.data[src.entity].feature,
            src.n_states,
            sigma_pos,
            sigma_rot,
        )

    def _summary_ruler(
        self, src: EntityNode, goal_value: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        assert isinstance(src, ValueNode)
        entity = src.con.entities[src.entity]
        model = src.con.models[src.entity]
        up = model.get_parameters()
        cov = entity.best_component_cov(goal_value, up)
        sigma_pos = np.maximum(np.sqrt(cov[:3]), np.exp(Entity.LSTD_FLOOR))
        if entity.cfg.add_rotation and cov.shape[0] >= 6:
            sigma_rot = np.maximum(np.sqrt(cov[3:6]), np.exp(Entity.LSTD_FLOOR))
        else:
            sigma_rot = None  # rotation ruler unavailable -> constant floor
        return sigma_pos, sigma_rot
