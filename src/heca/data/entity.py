import math
import warnings
from typing import Any

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from heca.misc.base import Configurable
from heca.data.data import DCEntity, DCScene
from heca.utils.quaternion import Quaternion
from heca.misc import logger


class Entity(Configurable):
    FEATURE_DIM: int = 33  # 16 (logits) + 13 (pose) + 2*2 (max extra for revolute)
    MAX_STATE_DIM: int = 16
    BASE_LOGSTD = -10.0
    LOGIT_CONFIDENCE = 10.0
    POS_DIM: int = 3
    ROT_DIM: int = 3
    ANCHOR_THRESHOLD: float = 0.1
    REG_COVAR = 1e-6
    LSTD_FLOOR = -2.0
    Z_CLIP = 10.0

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        type_id: int
        add_rotation: bool = True
        n_states: int = 1
        question: str = ""
        answers: list[str] = field(default_factory=list)
        max_fit_components: int = 10
        z_quantile_joint: float = 0.999
        z_quantile_dim: float = 0.999
        pos_sigma: float = 0.01
        rot_sigma: float = 0.01
        ext_sigma: float = 0.01
        canonicalize_rot: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def rot_dim(self) -> int:
        return Entity.ROT_DIM if self.cfg.add_rotation else 0

    def model_value(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if self.cfg.add_rotation:
            return value
        return np.concatenate(
            [value[..., : self.POS_DIM], value[..., self.POS_DIM + self.ROT_DIM :]],
            axis=-1,
        )

    def model_to_value(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        if self.cfg.add_rotation:
            return value
        zeros = np.zeros(value.shape[:-1] + (self.ROT_DIM,), dtype=value.dtype)
        return np.concatenate(
            [value[..., : self.POS_DIM], zeros, value[..., self.POS_DIM :]],
            axis=-1,
        )

    @property
    def measurement(self) -> dict:
        raise NotImplementedError

    def common_pose_part(self, label: str, obs: dict, normalize_pos=None) -> np.ndarray:
        pos = obs[f"heca_{label}_pos"]
        rot = obs[f"heca_{label}_rot"]
        if normalize_pos is not None:
            pos = normalize_pos(pos)
        # heca_*_rot is already (w, x, y, z); keep it before log-mapping.
        rot = np.array(rot, dtype=np.float32)
        quat = Quaternion.normalize(rot)
        aa = Quaternion.log_map(quat)
        return np.concatenate((pos, aa))

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def value_from_gt(self, label: str, obs: dict, normalize_pos=None) -> DCEntity:
        pose = self.common_pose_part(label, obs, normalize_pos=normalize_pos)
        extra = self.extra_part(label, obs)
        ste = obs[f"heca_{label}_ste"]
        return self.dc_from_parsed(pose, extra, ste)

    def dc_from_parsed(
        self, pose: np.ndarray, extra: np.ndarray, ste: np.ndarray
    ) -> DCEntity:
        value = np.concatenate(
            (
                np.asarray(pose).ravel(),
                np.asarray(extra).ravel(),
                np.asarray(ste).ravel(),
            )
        )
        feature = self.gnn_format(value)
        return DCEntity(value=value, feature=feature)

    def env_state_value(
        self, label: str, x: DCScene, unnormalize_pos=None
    ) -> dict[str, Any]:
        raise NotImplementedError

    def value_from_image(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def gnn_format(self, value: np.ndarray) -> np.ndarray:
        M = Entity.MAX_STATE_DIM
        D = len(value) - 7  # pos(3) + aa(3) + ste(1) = 7 base, extra is rest

        feat = np.zeros(Entity.FEATURE_DIM, dtype=np.float32)

        state_id = value[6 + D].astype(int)
        feat[: self.cfg.n_states] = -self.LOGIT_CONFIDENCE
        if state_id < self.cfg.n_states:
            feat[state_id] = self.LOGIT_CONFIDENCE

        feat[M : M + 3] = value[0:3]
        feat[M + 3 : M + 6] = self.BASE_LOGSTD

        quat = Quaternion.exp(value[3:6])
        feat[M + 6 : M + 10] = Quaternion.normalize(quat)
        feat[M + 10 : M + 13] = self.BASE_LOGSTD

        if D > 0:
            feat[M + 13 : M + 13 + D] = value[6 : 6 + D]
            feat[M + 13 + D : M + 13 + 2 * D] = self.BASE_LOGSTD

        return feat

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        raise NotImplementedError

    def sanitize_value(
        self,
        value: np.ndarray,
        lo: np.ndarray | None = None,
        hi: np.ndarray | None = None,
    ) -> np.ndarray:
        value = np.asarray(value).copy()
        if lo is not None or hi is not None:
            value = np.clip(value, lo, hi)
        return value

    def pose_sigma_variance(self) -> np.ndarray:
        n = int(self.measurement["pose"]["n_columns"])
        n_rot = self.rot_dim
        n_extra = n - Entity.POS_DIM - n_rot
        sigmas = np.concatenate(
            [
                np.full(Entity.POS_DIM, self.cfg.pos_sigma),
                np.full(n_rot, self.cfg.rot_sigma),
                np.full(n_extra, self.cfg.ext_sigma),
            ],
            dtype=float,
        )
        return sigmas**2

    def secure_mix_parameters(
        self, p: dict, eps: float = 1e-15, add_variance: bool = False
    ) -> dict:
        pis = p["measurement"]["state"]["pis"]
        n_outcomes = max(self.cfg.n_states, pis.shape[1])
        padded = np.full((pis.shape[0], n_outcomes), eps, dtype=np.float32)
        padded[:, : pis.shape[1]] = pis
        # Renormalize so probabilities sum to 1
        padded /= padded.sum(axis=1, keepdims=True)
        p["measurement"]["state"]["pis"] = padded
        if pis.shape[1] > self.cfg.n_states:
            warnings.warn(
                f"Fitted categorical has {pis.shape[1]} outcomes but "
                f"cfg.n_states={self.cfg.n_states}. Increase n_states in the "
                "entity config so the GNN features do not silently drop states.",
                stacklevel=2,
            )
        if add_variance:
            cov = p["measurement"]["pose"]["covariances"]
            p["measurement"]["pose"]["covariances"] = cov + self.pose_sigma_variance()
        return p

    def score_single(self, sample: np.ndarray, up: dict, eps: float = 1e-15) -> bool:
        sample = self.model_value(sample)
        p = self.secure_mix_parameters(up, add_variance=True)
        pose = sample[:-1]
        state = int(sample[-1])
        pis = p["measurement"]["state"]["pis"]

        best_k, z, zd = self._best_component(pose, p, eps=eps)
        chi_sqrt = float(math.sqrt(chi2.ppf(self.cfg.z_quantile_joint, len(pose))))

        valid_pose = z <= chi_sqrt and bool(np.all(zd <= self._z_dim_sigma))
        valid_state = bool(pis[best_k][state] > 1e-6)

        return valid_pose and valid_state

    def score_state(self, sample: np.ndarray, up: dict, eps: float = 1e-15) -> bool:
        sample = self.model_value(sample)
        p = self.secure_mix_parameters(up, add_variance=True)
        state = int(sample[-1])
        pis = p["measurement"]["state"]["pis"]

        best_k, z, zd = self._best_component(sample[:-1], p, eps=eps)
        return bool(pis[best_k][state] > 1e-6)

    @property
    def _z_dim_sigma(self) -> float:
        return float(norm.ppf(0.5 + self.cfg.z_quantile_dim / 2.0))

    def _best_component(
        self, pose: np.ndarray, p: dict, eps: float = 1e-15
    ) -> tuple[int, float, np.ndarray]:
        weights = p["weights"]
        means = p["measurement"]["pose"]["means"]
        vars_ = p["measurement"]["pose"]["covariances"]
        best_k, best_post = -1, -np.inf
        for k in range(len(weights)):
            var = np.maximum(vars_[k], eps)
            post = np.log(weights[k]) - 0.5 * np.sum(
                np.log(2 * np.pi * var) + (pose - means[k]) ** 2 / var
            )
            if post > best_post:
                best_k, best_post = k, post
        zd = np.abs(pose - means[best_k]) / np.sqrt(np.maximum(vars_[best_k], eps))
        z = float(np.sqrt(np.sum(zd**2)))
        return best_k, z, zd

    def best_component_cov(
        self, value: np.ndarray, up: dict, eps: float = 1e-15
    ) -> np.ndarray:
        sample = self.model_value(value)
        p = self.secure_mix_parameters(up, add_variance=True)
        pose = sample[:-1]
        best_k, _, _ = self._best_component(pose, p, eps=eps)
        return np.maximum(p["measurement"]["pose"]["covariances"][best_k], eps)

    def _ellipsoids_intersect(
        self,
        mu1: np.ndarray,
        var1: np.ndarray,
        mu2: np.ndarray,
        var2: np.ndarray,
        c: float,
        z_max: float | None = None,
        eps: float = 1e-15,
    ) -> bool:
        s1 = np.sqrt(np.maximum(var1, eps))
        s2 = np.sqrt(np.maximum(var2, eps))
        r = float(np.sqrt(c))
        if z_max is not None:
            r = min(r, z_max)

        # Fast accept: a component center inside the other ellipsoid (and caps)
        # is itself a common value.
        if np.sum((mu2 - mu1) ** 2 / np.maximum(var1, eps)) <= c and (
            z_max is None or bool(np.all(np.abs(mu2 - mu1) <= z_max * s1))
        ):
            return True
        if np.sum((mu1 - mu2) ** 2 / np.maximum(var2, eps)) <= c and (
            z_max is None or bool(np.all(np.abs(mu1 - mu2) <= z_max * s2))
        ):
            return True

        # Fast reject: per-dimension projections (and caps) are disjoint.
        lo = np.maximum(mu1 - r * s1, mu2 - r * s2)
        hi = np.minimum(mu1 + r * s1, mu2 + r * s2)
        if np.any(lo > hi):
            return False

        # Exact decider: convex min-max over the box.
        var1c = np.maximum(var1, eps)
        var2c = np.maximum(var2, eps)

        def g(x: np.ndarray) -> float:
            q1 = float(np.sum((x - mu1) ** 2 / var1c)) / c
            q2 = float(np.sum((x - mu2) ** 2 / var2c)) / c
            return max(q1, q2)

        x0 = np.clip((mu1 / var1c + mu2 / var2c) / (1.0 / var1c + 1.0 / var2c), lo, hi)
        res = minimize(
            g,
            x0,
            method="Nelder-Mead",
            bounds=[(float(lo[d]), float(hi[d])) for d in range(len(mu1))],
            options={"maxiter": 1000, "xatol": 1e-10, "fatol": 1e-12},
        )
        return float(res.fun) <= 1.0

    def containment(self, up1: dict, up2: dict, eps: float = 1e-15):
        p1 = self.secure_mix_parameters(up1)
        p2 = self.secure_mix_parameters(up2)
        w1 = p1["weights"]
        means1 = p1["measurement"]["pose"]["means"]
        vars1 = p1["measurement"]["pose"]["covariances"]
        pis1 = p1["measurement"]["state"]["pis"]
        w2 = p2["weights"]
        means2 = p2["measurement"]["pose"]["means"]
        vars2 = p2["measurement"]["pose"]["covariances"]
        pis2 = p2["measurement"]["state"]["pis"]

        d = means1.shape[1]
        chi = float(chi2.ppf(self.cfg.z_quantile_joint, d))

        def agrees(i: int, j: int) -> bool:
            var1 = np.maximum(vars1[i], eps)
            var2 = np.maximum(vars2[j], eps)
            # Exact ellipsoid-intersection test (instead of checking only the
            # posterior-mean agreement value, which is conservative when the
            # two variances differ a lot).
            if not self._ellipsoids_intersect(
                means1[i],
                var1,
                means2[j],
                var2,
                chi,
                self._z_dim_sigma,
            ):
                return False
            # Hard state gate: most likely states must be equal (aligned to the
            # union of observed states so the indices are comparable).
            n = max(len(pis1[i]), len(pis2[j]))
            c1 = np.pad(pis1[i], (0, n - len(pis1[i]))) if len(pis1[i]) < n else pis1[i]
            c2 = np.pad(pis2[j], (0, n - len(pis2[j]))) if len(pis2[j]) < n else pis2[j]
            return int(np.argmax(c1)) == int(np.argmax(c2))

        # Covered mass of each side: weight of components that have at least
        # one agreeing partner in the other distribution.
        covered1 = sum(
            w1[i] for i in range(len(w1)) if any(agrees(i, j) for j in range(len(w2)))
        )
        covered2 = sum(
            w2[j] for j in range(len(w2)) if any(agrees(i, j) for i in range(len(w1)))
        )
        return float(min(covered1, covered2))

    def containment_score(self, up1: dict, up2: dict) -> bool:
        """Decision wrapper: does a value exist that can occur under *both*
        conditions? Returns ``True`` iff ``containment`` is ``> 0`` and logs
        the actual float value for diagnostics."""
        value = self.containment(up1, up2)
        logger.debug(f"containment={value:.4f}")
        return value > 0.0

    def best_sample(self, up1: dict, up2: dict, eps: float = 1e-15):
        p1 = self.secure_mix_parameters(up1)
        p2 = self.secure_mix_parameters(up2)

        weights1 = p1["weights"]
        weights2 = p2["weights"]
        means1 = p1["measurement"]["pose"]["means"]
        means2 = p2["measurement"]["pose"]["means"]
        vars1 = p1["measurement"]["pose"]["covariances"]
        vars2 = p2["measurement"]["pose"]["covariances"]
        state1 = p1["measurement"]["state"]["pis"]
        state2 = p2["measurement"]["state"]["pis"]

        # Align categorical distributions to the union of observed states
        K1, K2 = len(weights1), len(weights2)
        results = []
        for i in range(K1):
            for j in range(K2):
                # Gaussian diag part
                precision = 1.0 / vars1[i] + 1.0 / vars2[j]
                var = 1.0 / precision
                mean = var * (means1[i] / vars1[i] + means2[j] / vars2[j])
                diff = means1[i] - means2[j]
                log_norm = -0.5 * (
                    np.sum(np.log(2 * np.pi * (vars1[i] + vars2[j])))
                    + np.sum(diff**2 / (vars1[i] + vars2[j]))
                )
                # Categorical part — padded to the union of observed states
                n = max(len(state1[i]), len(state2[j]))
                c1 = (
                    np.pad(state1[i], (0, n - len(state1[i])))
                    if len(state1[i]) < n
                    else state1[i]
                )
                c2 = (
                    np.pad(state2[j], (0, n - len(state2[j])))
                    if len(state2[j]) < n
                    else state2[j]
                )
                cat_prod = c1 * c2  # element-wise over aligned states
                state = int(np.argmax(cat_prod))
                if cat_prod[state] <= 0.0:
                    # Disjoint categorical supports: pick the state with the
                    # highest combined mass instead of an arbitrary argmax(0).
                    state = int(np.argmax(c1 + c2))
                log_cat = np.log(np.clip(cat_prod[state], eps, None))
                score = np.log(weights1[i]) + np.log(weights2[j]) + log_norm + log_cat
                results.append({"score": score, "pose": mean, "state": state})

        results.sort(key=lambda r: r["score"], reverse=True)
        pose = results[0]["pose"]
        state = results[0]["state"]
        assert isinstance(pose, np.ndarray)
        return self.model_to_value(np.concatenate([pose, [state]]))

    def comp_feature(
        self, up: dict, eps: float = 1e-8
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        NOTE: ASSUMES MODELS USE DIAG MODE
        Returns:
            np.ndarray of shape (N, FEATURE_DIM) with layout:
                - [0:MAX_STATE_DIM]             = state scores on the same
                  [-LOGIT_CONFIDENCE, +LOGIT_CONFIDENCE] scale as gnn_format
                  (2*p - 1) * LOGIT_CONFIDENCE; impossible states stay at -10
                - [M:M+3]                       = μ_pos
                - [M+3:M+6]                     = log(σ_pos)
                - [M+6:M+10]                    = quaternion [w, x, y, z]
                - [M+10:M+13]                   = log(σ_rot, tangent space)
                - [M+13:M+13+D]                 = μ_extra
                - [M+13+D:M+13+2D]              = log(σ_extra)
            where M = MAX_STATE_DIM, D = n_columns - 6
        """

        p = self.secure_mix_parameters(up)
        means = p["measurement"]["pose"]["means"]  # (N, n_columns)
        covariances = p["measurement"]["pose"]["covariances"]  # (N, n_columns)
        pis = p["measurement"]["state"]["pis"]  # (N, K)
        N = len(p["weights"])
        M = Entity.MAX_STATE_DIM
        base = Entity.POS_DIM + (Entity.ROT_DIM if self.cfg.add_rotation else 0)
        D = means.shape[1] - base  # extra continuous dims beyond pos (+rot)

        feat = np.zeros((N, Entity.FEATURE_DIM), dtype=np.float32)

        # State scores — same convention and scale as gnn_format.
        feat[:, : self.cfg.n_states] = -self.LOGIT_CONFIDENCE
        n_logit = min(pis.shape[1], self.cfg.n_states)
        feat[:, :n_logit] = self.LOGIT_CONFIDENCE * (2.0 * pis[:, :n_logit] - 1.0)

        # Pose: position mean + logstd
        feat[:, M : M + 3] = means[:, 0:3]
        feat[:, M + 3 : M + 6] = 0.5 * np.log(covariances[:, 0:3] + eps)

        # Pose: axis-angle -> quaternion + rotation logstd (3 dof in tangent space)
        if self.cfg.add_rotation:
            quat = Quaternion.exp(means[:, 3:6])
            feat[:, M + 6 : M + 10] = Quaternion.normalize(quat)
            feat[:, M + 10 : M + 13] = 0.5 * np.log(covariances[:, 3:6] + eps)
        else:
            feat[:, M + 6 : M + 10] = Quaternion.identity()
            feat[:, M + 10 : M + 13] = self.BASE_LOGSTD

        # Extra continuous dims: mean + logstd
        if D > 0:
            feat[:, M + 13 : M + 13 + D] = means[:, base:]
            feat[:, M + 13 + D : M + 13 + 2 * D] = 0.5 * np.log(
                covariances[:, base:] + eps
            )

        return feat, p["weights"]

    # def kl_variational_paper(self, other: "Condition", key: str):
    #     m1, s1 = Condition.mix_with_states(self, key)
    #     m2, s2 = Condition.mix_with_states(other, key)
    #     p1 = m1.get_parameters()
    #     p2 = m2.get_parameters()

    #     # Align categorical distributions to a common state space
    #     if s1 is not None and s2 is not None:
    #         target = sorted(s1 | s2)  # union
    #         padded_cat1 = self._pad_cat_probs(
    #             p1["measurement"]["state"]["pis"], s1, target
    #         )
    #         padded_cat2 = self._pad_cat_probs(
    #             p2["measurement"]["state"]["pis"], s2, target
    #         )
    #     else:
    #         padded_cat1 = p1["measurement"]["state"]["pis"]
    #         padded_cat2 = p2["measurement"]["state"]["pis"]

    #     kl = 0.0
    #     for i in range(len(p1["weights"])):
    #         w_i = p1["weights"][i]
    #         mu1 = p1["measurement"]["pose"]["means"][i]
    #         var1 = p1["measurement"]["pose"]["covariances"][i]
    #         cat1 = padded_cat1[i]

    #         # Numerator: self-overlap of component i with its OWN model
    #         log_sum_m1 = 0.0
    #         for k in range(len(p1["weights"])):
    #             w_k = p1["weights"][k]
    #             mu_k = p1["measurement"]["pose"]["means"][k]
    #             var_k = p1["measurement"]["pose"]["covariances"][k]
    #             cat_k = padded_cat1[k]

    #             kl_self = 0.5 * np.sum(
    #                 np.log(var1)
    #                 - np.log(var_k)
    #                 + var_k / var1
    #                 + (mu_k - mu1) ** 2 / var1
    #                 - 1
    #             )
    #             cat1_safe = np.clip(cat1, 1e-12, 1)
    #             cat_k_safe = np.clip(cat_k, 1e-12, 1)
    #             kl_cat_self = np.sum(
    #                 cat1_safe * (np.log(cat1_safe) - np.log(cat_k_safe))
    #             )
    #             log_sum_m1 += w_k * np.exp(-(kl_self + kl_cat_self))

    #         # Denominator: cross-overlap with model 2
    #         log_sum_m2 = 0.0
    #         for j in range(len(p2["weights"])):
    #             w_j = p2["weights"][j]
    #             mu2 = p2["measurement"]["pose"]["means"][j]
    #             var2 = p2["measurement"]["pose"]["covariances"][j]
    #             cat2 = padded_cat2[j]

    #             kl_gauss = 0.5 * np.sum(
    #                 np.log(var2)
    #                 - np.log(var1)
    #                 + var1 / var2
    #                 + (mu1 - mu2) ** 2 / var2
    #                 - 1
    #             )
    #             cat2_safe = np.clip(cat2, 1e-12, 1)
    #             kl_cat = np.sum(cat1_safe * (np.log(cat1_safe) - np.log(cat2_safe)))
    #             log_sum_m2 += w_j * np.exp(-(kl_gauss + kl_cat))

    #         kl += w_i * np.log(log_sum_m1 / log_sum_m2)
    #         if key == "window_handle":
    #             print(f"Component {i}:")
    #             print(f"mu1={mu1}")
    #             print(f"mu2={mu2}")
    #             print(f"var1={var1}")
    #             print(f"var2={var2}")
    #             print(f"cat1={cat1}, cat2={cat2}")
    #             print(f"log_sum_m1={log_sum_m1:.4f}")
    #             print(f"log_sum_m2={log_sum_m2:.4f}")
    #     if key == "window_handle":
    #         print(f"Total KL={kl:.4f}, score={np.exp(-kl):.4f}")
    #     return np.exp(-kl)
