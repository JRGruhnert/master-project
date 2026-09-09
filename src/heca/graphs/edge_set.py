from typing import Generic, TypeVar

from enum import Enum

import numpy as np
import torch

from heca.graphs.node_set import NodeSet
from heca.graphs.node import CompNode, EntityNode, GraphNode, OptionNode
from heca.data.entity import Entity
from heca.utils.quaternion import Quaternion

S = TypeVar("S", bound=GraphNode)
D = TypeVar("D", bound=GraphNode)

LSTD_FLOOR = -2.0
Z_CLIP = 10.0


class ResidualMode(Enum):
    CONSTANT = "constant"
    ENTITY = "entity"
    POST = "post"


class EdgeSet(Generic[S, D]):
    def __init__(
        self,
        type: tuple[str, str, str],
        residual_mode: ResidualMode = ResidualMode.CONSTANT,
    ):
        self.edge_index: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edge_attr: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
        self.edges: list[tuple[int, int]] = []
        self.attrs: list[np.ndarray] = []
        self.rebuild: bool = True
        self.type = type
        self.residual_mode = residual_mode

    def add(self, src_idx: int, dst_idx: int):
        self.edges.append((src_idx, dst_idx))
        self.attrs.append(np.zeros(0))
        self.rebuild = True

    def size(self) -> int:
        return len(self.edges)

    def build(self, snset: NodeSet[S], dnset: NodeSet[D]):
        src_list, dst_list = zip(*self.edges)
        self.edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        for i, edge in enumerate(self.edges):
            src = snset.idx_get(edge[0])
            dst = dnset.idx_get(edge[1])
            self.update_attr(snset, src, dst, i)
        self.edge_attr = torch.from_numpy(np.stack(self.attrs)).float()

    def update_attr(self, snset: NodeSet[S], src: S, dst: D, index: int):
        if self.type == ("entity", "stepmix", "entity"):
            assert isinstance(src, CompNode)
            assert isinstance(dst, EntityNode)
            assert src.n_states == dst.n_states, "Sanity check"
            self.attrs[index] = self.stepmix_feat(
                src.data.feature, dst.data.feature, src.weight, src.n_states
            )
        elif self.type == ("entity", "summary", "option"):
            assert isinstance(src, EntityNode)
            assert isinstance(dst, OptionNode)
            goal_value = None
            try:
                goal_value = dst.data[src.entity].value
            except (KeyError, AttributeError):
                pass  # set_goal not called yet -> fall back to constant floor
            sigma_pos, sigma_rot = self._summary_ruler(snset, src, goal_value)
            self.attrs[index] = self.summary_feat(
                src.data.feature,
                dst.data[src.entity].feature,
                src.n_states,
                sigma_pos,
                sigma_rot,
            )
        elif self.type == ("entity", "tapas", "entity"):
            self.attrs[index] = np.empty(1)
        else:
            raise NotImplementedError

    def _summary_ruler(
        self,
        snset: NodeSet,
        src: EntityNode,
        goal_value: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        O = Entity.MAX_STATE_DIM
        if self.residual_mode == ResidualMode.CONSTANT:
            return None, None

        if self.residual_mode == ResidualMode.POST:
            # B2: best-matching post component w.r.t. the goal value.
            con = getattr(src, "con", None)
            if con is None or goal_value is None:
                return None, None  # chain sub-nodes carry no Condition
            entity = con.entities.get(src.entity)
            model = con.models.get(src.entity)
            if entity is None or model is None:
                return None, None
            try:
                up = model.get_parameters()
                cov = entity.best_component_cov(goal_value, up)
            except Exception:
                return None, None  # fall back to the constant floor
            sigma_pos = np.maximum(np.sqrt(cov[:3]), np.exp(LSTD_FLOOR))
            if entity.cfg.add_rotation and cov.shape[0] >= 6:
                sigma_rot = np.maximum(np.sqrt(cov[3:6]), np.exp(LSTD_FLOOR))
            else:
                sigma_rot = None  # rotation ruler unavailable -> constant floor
            return sigma_pos, sigma_rot

        # ENTITY: every comp node of this entity in the graph
        comps: list[CompNode] = []
        for node in snset.items:
            if isinstance(node, CompNode) and node.entity == src.entity:
                comps.append(node)

        lstd_pos, lstd_rot = [], []
        for c in comps:
            f = c.data.feature
            if f.shape[0] > O + 12:
                lstd_pos.append(f[O + 3 : O + 6])
                lstd_rot.append(f[O + 10 : O + 13])
        if not lstd_pos:
            return None, None  # fall back to the constant floor

        # Robust per-dim ruler: median fitted log-std -> sigma, floored.
        med_pos = np.median(np.stack(lstd_pos), axis=0)
        med_rot = np.median(np.stack(lstd_rot), axis=0)
        sigma_pos = np.maximum(np.exp(med_pos), np.exp(LSTD_FLOOR))
        sigma_rot = np.maximum(np.exp(med_rot), np.exp(LSTD_FLOOR))
        return sigma_pos, sigma_rot

    def stepmix_feat(
        self, x_src: np.ndarray, x_dst: np.ndarray, w_src: float, n_states: int
    ) -> np.ndarray:
        feat = self.residual(x_src, x_dst, n_states)
        return np.concatenate([feat, [w_src]])

    def summary_feat(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        n_states: int,
        sigma_pos: np.ndarray | None = None,
        sigma_rot: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.residual(
            x_src, x_dst, n_states, sigma_pos=sigma_pos, sigma_rot=sigma_rot
        )

    def tapas_feat(self, x_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
        return np.empty(1)  # We dont have edge features

    def residual(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        n_states: int,
        sigma_pos: np.ndarray | None = None,
        sigma_rot: np.ndarray | None = None,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """
        Compute directed edge features from source nodes to destination nodes.

        Args:
            x_src: [feature_dim] features for src nodes
            x_dst: [feature_dim] features for dst nodes
            sigma_pos / sigma_rot: optional per-dimension rulers [3]. When given
                the residual divides the raw delta by them (relative scale
                supplied by the caller, e.g. entity/post fitted sigma). When
                None the residual uses the nodes' own log-stds (floored).

        Returns:
            edge_feat: [8] normalized residuals (z_pos + z_rot + z_state + w_src)
        """
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
            lstd_pos_a = np.maximum(lstd_pos_a, LSTD_FLOOR)
            lstd_pos_b = np.maximum(lstd_pos_b, LSTD_FLOOR)
            var_pos_a = np.exp(2 * lstd_pos_a)
            var_pos_b = np.exp(2 * lstd_pos_b)
            var_comb_pos = var_pos_a + var_pos_b
            z_pos = (mu_pos_b - mu_pos_a) / np.sqrt(var_comb_pos + eps)
        else:
            z_pos = (mu_pos_b - mu_pos_a) / (sigma_pos + eps)

        if sigma_rot is None:
            lstd_rot_a = np.maximum(lstd_rot_a, LSTD_FLOOR)
            lstd_rot_b = np.maximum(lstd_rot_b, LSTD_FLOOR)
            var_rot_a = np.exp(2 * lstd_rot_a)
            var_rot_b = np.exp(2 * lstd_rot_b)
            var_comb_rot = var_rot_a + var_rot_b
            z_rot = r_vec / np.sqrt(var_comb_rot + eps)
        else:
            z_rot = r_vec / (sigma_rot + eps)

        z_pos = np.clip(z_pos, -Z_CLIP, Z_CLIP)
        z_rot = np.clip(z_rot, -Z_CLIP, Z_CLIP)

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
            f"{self.size()} edges, attr.shape={self.edge_attr.shape}"
        )
