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
    def __init__(self):
        self.edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edges: list[tuple[int, int]] = []
        self.attrs: list[np.ndarray] = []
        self.rebuild: bool = True

    def add(self, src_idx: int, dst_idx: int):
        self.edges.append((src_idx, dst_idx))
        self.attrs.append(np.zeros(0))
        self.rebuild = True

    @property
    def size(self) -> int:
        return len(self.edges)

    def build(self, snset: NodeSet[S], dnset: NodeSet[D]):
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
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
        n_states: int,
        sigma_pos: np.ndarray | None = None,
        sigma_rot: np.ndarray | None = None,
        eps: float = 1e-15,
    ) -> np.ndarray:

        O = Entity.MAX_STATE_DIM

        # Unpack source (A)
        logits_a = x_src[:n_states]
        mu_pos_a = x_src[O : O + 3]
        lstd_pos_a = x_src[O + 3 : O + 6]
        q_a = x_src[O + 6 : O + 10]
        lstd_rot_a = x_src[O + 10 : O + 13]

        # Unpack destination (B)
        logits_b = x_dst[:n_states]
        mu_pos_b = x_dst[O : O + 3]
        lstd_pos_b = x_dst[O + 3 : O + 6]
        q_b = x_dst[O + 6 : O + 10]
        lstd_rot_b = x_dst[O + 10 : O + 13]

        q_inv = Quaternion.inv(q_a)
        q_rel = Quaternion.mul(q_b, q_inv)
        r_vec = Quaternion.log_map(q_rel)  # [E, 3]

        if sigma_pos is None:
            lstd_pos_a = np.maximum(lstd_pos_a, Entity.LSTD_FLOOR)
            lstd_pos_b = np.maximum(lstd_pos_b, Entity.LSTD_FLOOR)
            var_pos_a = np.exp(2 * lstd_pos_a)
            var_pos_b = np.exp(2 * lstd_pos_b)
            var_comb_pos = var_pos_a + var_pos_b
            z_pos = (mu_pos_b - mu_pos_a) / np.sqrt(var_comb_pos + eps)
        else:
            z_pos = (mu_pos_b - mu_pos_a) / (sigma_pos + eps)

        if sigma_rot is None:
            lstd_rot_a = np.maximum(lstd_rot_a, Entity.LSTD_FLOOR)
            lstd_rot_b = np.maximum(lstd_rot_b, Entity.LSTD_FLOOR)
            var_rot_a = np.exp(2 * lstd_rot_a)
            var_rot_b = np.exp(2 * lstd_rot_b)
            var_comb_rot = var_rot_a + var_rot_b
            z_rot = r_vec / np.sqrt(var_comb_rot + eps)
        else:
            z_rot = r_vec / (sigma_rot + eps)

        z_pos = np.clip(z_pos, -Entity.Z_CLIP, Entity.Z_CLIP)
        z_rot = np.clip(z_rot, -Entity.Z_CLIP, Entity.Z_CLIP)

        logits_a_max = np.max(logits_a, axis=-1, keepdims=True)
        logits_b_max = np.max(logits_b, axis=-1, keepdims=True)
        softmax_a = np.exp(logits_a - logits_a_max) / np.sum(
            np.exp(logits_a - logits_a_max), axis=-1, keepdims=True
        )
        softmax_b = np.exp(logits_b - logits_b_max) / np.sum(
            np.exp(logits_b - logits_b_max), axis=-1, keepdims=True
        )

        # Cross-entropy
        z_state = -np.sum(softmax_b * np.log(softmax_a + eps), axis=-1)

        # Clip to prevent extreme outliers from dominating the edge feature
        z_state = np.clip(z_state, 0.0, 10.0)

        return np.concatenate([z_pos, z_rot, np.atleast_1d(z_state)], axis=-1)
