from typing import Generic, TypeVar

import numpy as np
import torch

from heca.data.entity import Entity
from heca.graphs.nodes.node_set import NodeSet
from heca.graphs.nodes.node import GraphNode
from heca.utils.quaternion import Quaternion

S = TypeVar("S", bound=GraphNode)
D = TypeVar("D", bound=GraphNode)


class EdgeSet(Generic[S, D]):
    has_attrs: bool = True

    def __init__(self):
        self.edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edges: list[tuple[int, int]] = []
        self.attrs: list[np.ndarray] = []
        self.rebuild: bool = True

    def add(self, src_idx: int, dst_idx: int):
        self.edges.append((src_idx, dst_idx))
        if self.has_attrs:
            self.attrs.append(np.zeros(0))
        self.rebuild = True

    @property
    def size(self) -> int:
        return len(self.edges)

    @staticmethod
    def gather_features(nset: NodeSet, indices) -> np.ndarray:
        assert nset.x.shape[0] == len(nset.items), f"{nset.type} node set not built"
        # np.asarray: a tuple of ints would index as multiple dimensions.
        return nset.x.numpy()[np.asarray(indices, dtype=np.intp)]

    def build(self, snset: NodeSet[S], dnset: NodeSet[D]):
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        if not self.has_attrs:
            self.edge_attr = torch.empty((2, 0), dtype=torch.long)
            return
        for i, edge in enumerate(self.edges):
            src = snset.idx_get(edge[0])
            dst = dnset.idx_get(edge[1])
            self.update_attr(src, dst, i)
        self.edge_attr = torch.from_numpy(np.stack(self.attrs)).float()

    def update_attr(self, src: S, dst: D, index: int):
        raise NotImplementedError

    @property
    def type(self) -> tuple[str, str, str]:
        raise NotImplementedError

    def edges_from_sets(self, snset: NodeSet[S], tnset: NodeSet[D], src_key: str):
        """Create edges by matching node source entries to this edge type."""
        for i, node in enumerate(tnset.items):
            for key in node.sources.get(src_key, set()):
                if snset.has_key(key):
                    j = snset.get_index(key)
                    self.add(j, i)

    def __str__(self) -> str:
        return (
            f"EdgeSet({self.type[0]}→{self.type[2]}): "
            f"{self.size} edges, attr.shape={self.edge_attr.shape}"
        )

    def residual(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
    ) -> np.ndarray:
        """Edge feature for a single src/dst feature pair (``[F]`` each)."""
        return self.residual_batch(x_src, x_dst)[0]

    @staticmethod
    def residual_batch(x_src: np.ndarray, x_dst: np.ndarray, eps=1e-15) -> np.ndarray:
        O = Entity.MAX_STATE_DIM
        x_src = np.atleast_2d(x_src)
        x_dst = np.atleast_2d(x_dst)

        mu_pos_a = x_src[:, O : O + 3]
        q_a = x_src[:, O + 6 : O + 10]
        mu_pos_b = x_dst[:, O : O + 3]
        q_b = x_dst[:, O + 6 : O + 10]

        r_vec = Quaternion.log_map(Quaternion.mul(q_b, Quaternion.inv(q_a)))

        z_pos = np.clip(mu_pos_b - mu_pos_a, -Entity.Z_CLIP, Entity.Z_CLIP)
        z_rot = np.clip(r_vec, -Entity.Z_CLIP, Entity.Z_CLIP)

        # Cross-entropy between the two state distributions.
        p_src = x_src[:, :O]
        p_dst = x_dst[:, :O]
        z_state = -np.sum(p_dst * np.log(p_src + eps), axis=-1)

        # Clip to prevent extreme outliers from dominating the edge feature
        z_state = np.clip(z_state, 0.0, Entity.Z_CLIP)

        return np.concatenate([z_pos, z_rot, z_state[:, None]], axis=-1)
