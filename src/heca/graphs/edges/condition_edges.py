import numpy as np
import torch

from heca.graphs.edges.edge_set import EdgeSet
from heca.graphs.nodes.node import CompNode, EntityNode
from heca.graphs.nodes.node_set import NodeSet


class ConditionEdges(EdgeSet[EntityNode, EntityNode]):

    @property
    def type(self) -> tuple[str, str, str]:
        return ("entity", "condition", "entity")

    def build(self, snset: NodeSet[EntityNode], dnset: NodeSet[EntityNode]):
        """Score every condition edge in one batched numpy pass.

        The generic :meth:`EdgeSet.build` calls ``update_attr`` once per edge,
        and each of those calls pays the full small-array numpy overhead of
        ``residual`` (~10 ms per rebuild on scene0, i.e. per option step).
        """
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        x_src = self.gather_features(snset, src_list)
        x_dst = self.gather_features(dnset, dst_list)
        n_src = np.fromiter(
            (snset.items[i].n_states for i in src_list),  # type: ignore[attr-defined]
            dtype=np.int64,
            count=len(src_list),
        )
        n_dst = np.fromiter(
            (dnset.items[i].n_states for i in dst_list),
            dtype=np.int64,
            count=len(dst_list),
        )
        assert np.array_equal(n_src, n_dst), "Sanity check"
        weights = np.fromiter(
            (snset.items[i].weight for i in src_list),  # type: ignore[attr-defined]
            dtype=np.float32,
            count=len(src_list),
        )

        attr = np.concatenate(
            [self.residual_batch(x_src, x_dst, n_src), weights[:, None]], axis=-1
        )
        self.attrs = list(attr)
        self.edge_attr = torch.from_numpy(attr).float()

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
