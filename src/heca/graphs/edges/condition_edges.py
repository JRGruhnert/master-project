import numpy as np

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import CompNode, EntityNode


class ConditionEdges(EdgeSet[EntityNode, EntityNode]):

    @property
    def type(self) -> tuple[str, str, str]:
        return ("entity", "condition", "entity")

    def update_attr(self, src: CompNode, dst: EntityNode, index: int):
        assert src.n_states == dst.n_states, "Sanity check"
        self.attrs[index] = self.stepmix_feat(
            src.data.feature, dst.data.feature, src.weight, src.n_states
        )

    def stepmix_feat(
        self, x_src: np.ndarray, x_dst: np.ndarray, w_src: float, n_states: int
    ) -> np.ndarray:
        feat = self.residual(x_src, x_dst, n_states)
        return np.concatenate([feat, [w_src]])
