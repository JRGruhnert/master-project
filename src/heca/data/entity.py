import math
import warnings
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.stats import chi2, norm

from heca.misc.base import Configurable
from heca.data.data import DCEntity, DCScene
from heca.utils.quaternion import Quaternion
from heca.misc import logger


@lru_cache(maxsize=256)
def _chi2_ppf(q: float, df: int) -> float:
    return float(chi2.ppf(q, df))


@lru_cache(maxsize=256)
def _chi_sqrt(q: float, df: int) -> float:
    return float(math.sqrt(chi2.ppf(q, df)))


@lru_cache(maxsize=8)
def _z_dim_sigma(z_quantile_dim: float) -> float:
    return float(norm.ppf(0.5 + z_quantile_dim / 2.0))


@dataclass(frozen=True)
class FeatureBlock:
    start: int
    mean_dim: int
    logstd_dim: int = 0

    @property
    def dim(self) -> int:
        return self.mean_dim + self.logstd_dim

    def _live(self, offset: int, width: int, live: int | None) -> slice:
        return slice(
            self.start + offset, self.start + offset + (width if live is None else live)
        )

    def mean(self, live: int | None = None) -> slice:
        """The mean slots; ``live`` narrows to the dims an entity actually uses."""
        return self._live(0, self.mean_dim, live)

    def logstd(self, live: int | None = None) -> slice:
        return self._live(self.mean_dim, self.logstd_dim, live)


def _layout(
    max_state: int, max_extra: int, pos_dim: int, rot_dim: int
) -> dict[str, FeatureBlock]:
    blocks: dict[str, FeatureBlock] = {}
    offset = 0
    for name, mean_dim, logstd_dim in (
        ("state", max_state, 0),
        ("pos", pos_dim, rot_dim),
        ("rot", 4, rot_dim),
        ("extra", max_extra, max_extra),
    ):
        blocks[name] = FeatureBlock(offset, mean_dim, logstd_dim)
        offset += mean_dim + logstd_dim
    return blocks


def _feature_dim(layout: dict[str, FeatureBlock]) -> int:
    """Total width of a layout, and a check that its blocks are contiguous."""
    end = 0
    for name, block in layout.items():
        if block.start != end:
            raise ValueError(
                f"layout block '{name}' starts at {block.start}, expected {end} "
                "— the blocks must tile the feature vector"
            )
        end += block.dim
    return end


class Entity(Configurable):
    TYPE_NAMES: ClassVar[tuple[str, ...]] = (
        "free",
        "static",
        "prismatic",
        "revolute",
    )

    # Which measurement blocks a type has, in feature order. Each subclass sets
    # it and that type's encoder reads it from here, so it is written down once.
    BLOCKS: ClassVar[tuple[str, ...]] = ("state", "pos")

    MAX_STATE_DIM: int = 8
    MAX_EXTRA_DIM: int = 2
    BASE_LOGSTD = -10.0
    POS_DIM: int = 3
    ROT_DIM: int = 3
    ANCHOR_THRESHOLD: float = 0.1
    REG_COVAR = 1e-6
    Z_CLIP = 10.0

    LAYOUT: ClassVar[dict[str, FeatureBlock]] = _layout(
        MAX_STATE_DIM, MAX_EXTRA_DIM, POS_DIM, ROT_DIM
    )
    FEATURE_DIM: ClassVar[int] = _feature_dim(LAYOUT)

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
        pos_sigma: float = 0.01  # metres
        rot_sigma: float = 0.05  # radians of tangent-space rotation (~2.9 deg)
        sca_sigma: float = 0.05  # prismatic: fraction of the slide range
        ang_sigma: float = 0.05  # revolute: radians, as a chord on the unit circle
        ext_sigma: float = 0.05  # fallback for any other extra dims
        canonicalize_rot: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        unknown = set(self.BLOCKS) - set(Entity.LAYOUT)
        if unknown:
            raise ValueError(
                f"{type(self).__name__}.BLOCKS names unknown blocks "
                f"{sorted(unknown)}; the layout has {sorted(Entity.LAYOUT)}"
            )

        expected = Entity.TYPE_NAMES[cfg.type_id]
        actual = type(self).__name__.removesuffix("Entity").lower()
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__} carries type_id={cfg.type_id}, which "
                f"Entity.TYPE_NAMES maps to '{expected}'"
            )

    @property
    def rot_dim(self) -> int:
        return Entity.ROT_DIM if self.cfg.add_rotation else 0

    @property
    def pose_dim(self) -> int:
        return int(self.measurement["pose"]["n_columns"])

    @property
    def extra_sigma(self) -> np.ndarray:
        n = self.pose_dim - Entity.POS_DIM - self.rot_dim
        return np.full(n, self.cfg.ext_sigma)

    @property
    def pose_dof_groups(self) -> list[np.ndarray]:
        return [np.array([i]) for i in range(self.pose_dim)]

    @property
    def pose_dof(self) -> int:
        return len(self.pose_dof_groups)

    def group_squared(self, values: np.ndarray) -> np.ndarray:
        return np.array([float(np.sum(values[g] ** 2)) for g in self.pose_dof_groups])

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

    @staticmethod
    def _check_state_width(n_states: int) -> None:
        if n_states > Entity.MAX_STATE_DIM:
            raise ValueError(
                f"n_states={n_states} exceeds Entity.MAX_STATE_DIM="
                f"{Entity.MAX_STATE_DIM}; the state block would overwrite the "
                "pose blocks. Raise MAX_STATE_DIM (and "
                "Network.Config.max_state) to widen the layout."
            )

    def gnn_format(self, value: np.ndarray) -> np.ndarray:
        D = len(value) - 7  # pos(3) + aa(3) + ste(1) = 7 base, extra is rest
        if D > Entity.MAX_EXTRA_DIM:
            raise ValueError(
                f"value has {D} extra dims but the feature layout reserves "
                f"Entity.MAX_EXTRA_DIM={Entity.MAX_EXTRA_DIM}; raise that (the "
                "layout and feature width follow) to widen the graph features"
            )

        L = Entity.LAYOUT
        feat = np.zeros(Entity.FEATURE_DIM, dtype=np.float32)

        self._check_state_width(self.cfg.n_states)
        state_id = value[6 + D].astype(int)
        state = feat[L["state"].mean(self.cfg.n_states)]  # a view, writes through
        state[:] = 1.0 / self.cfg.n_states
        if 0 <= state_id < self.cfg.n_states:
            state[:] = 0.0
            state[state_id] = 1.0

        feat[L["pos"].mean()] = value[0:3]
        feat[L["pos"].logstd()] = self.BASE_LOGSTD

        if self.cfg.add_rotation:
            quat = Quaternion.exp(value[3:6])
            feat[L["rot"].mean()] = Quaternion.normalize(quat)
        else:
            feat[L["rot"].mean()] = Quaternion.identity()
        feat[L["rot"].logstd()] = self.BASE_LOGSTD

        if D > 0:
            feat[L["extra"].mean(D)] = value[6 : 6 + D]
            feat[L["extra"].logstd(D)] = self.BASE_LOGSTD

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
        sigmas = np.concatenate(
            [
                np.full(Entity.POS_DIM, self.cfg.pos_sigma),
                np.full(self.rot_dim, self.cfg.rot_sigma),
                self.extra_sigma,
            ],
            dtype=float,
        )
        assert len(sigmas) == self.pose_dim, (
            f"tolerance block ({len(sigmas)}) does not match the fitted pose "
            f"width ({self.pose_dim})"
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
        if pis.shape[1] > self.cfg.n_states:
            warnings.warn(
                f"Fitted categorical has {pis.shape[1]} outcomes but "
                f"cfg.n_states={self.cfg.n_states}. Increase n_states in the "
                "entity config so the GNN features do not silently drop states.",
                stacklevel=2,
            )

        measurement = dict(p["measurement"])
        measurement["state"] = {**measurement["state"], "pis": padded}
        if add_variance:
            pose = dict(measurement["pose"])
            pose["covariances"] = (
                np.asarray(pose["covariances"]) + self.pose_sigma_variance()
            )
            measurement["pose"] = pose
        return {**p, "measurement": measurement}

    def score_single(self, sample: np.ndarray, up: dict, eps: float = 1e-15) -> bool:
        return self.score_prepared(
            sample, self.secure_mix_parameters(up, add_variance=True), eps=eps
        )

    def prepare_single(self, up: dict) -> dict:
        return self.secure_mix_parameters(up, add_variance=True)

    def score_prepared(self, sample: np.ndarray, p: dict, eps: float = 1e-15) -> bool:
        """Same test as :meth:`score_single`, on already prepared parameters."""
        sample = self.model_value(sample)
        pose = sample[:-1]
        state = int(sample[-1])
        pis = p["measurement"]["state"]["pis"]

        best_k, z, zd = self._best_component(pose, p, eps=eps)
        chi_sqrt = _chi_sqrt(self.cfg.z_quantile_joint, self.pose_dof)
        capped = bool(np.all(self.group_squared(zd) <= self._z_dim_sigma**2))

        if not (z <= chi_sqrt and capped):
            return False
        return bool(pis[best_k][state] > 1e-6)

    def score_state(self, sample: np.ndarray, up: dict, eps: float = 1e-15) -> bool:
        sample = self.model_value(sample)
        p = self.secure_mix_parameters(up, add_variance=True)
        state = int(sample[-1])
        pis = p["measurement"]["state"]["pis"]

        best_k, z, zd = self._best_component(sample[:-1], p, eps=eps)
        return bool(pis[best_k][state] > 1e-6)

    @property
    def _z_dim_sigma(self) -> float:
        return _z_dim_sigma(self.cfg.z_quantile_dim)

    def _best_component(
        self, pose: np.ndarray, p: dict, eps: float = 1e-15
    ) -> tuple[int, float, np.ndarray]:
        weights = np.asarray(p["weights"])
        means = np.asarray(p["measurement"]["pose"]["means"])
        vars_ = np.asarray(p["measurement"]["pose"]["covariances"])
        var = np.maximum(vars_, eps)
        post = np.log(weights) - 0.5 * np.sum(
            np.log(2 * np.pi * var) + (pose - means) ** 2 / var, axis=-1
        )
        best_k = int(np.argmax(post))
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

        if np.sum((mu2 - mu1) ** 2 / np.maximum(var1, eps)) <= c and self._within_caps(
            mu2 - mu1, s1, z_max
        ):
            return True
        if np.sum((mu1 - mu2) ** 2 / np.maximum(var2, eps)) <= c and self._within_caps(
            mu1 - mu2, s2, z_max
        ):
            return True

        lo = np.maximum(mu1 - r * s1, mu2 - r * s2)
        hi = np.minimum(mu1 + r * s1, mu2 + r * s2)
        if np.any(lo > hi):
            return False

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

    def _within_caps(
        self, delta: np.ndarray, sigma: np.ndarray, z_max: float | None
    ) -> bool:
        if z_max is None:
            return True
        zd = np.abs(delta) / np.maximum(sigma, 1e-15)
        return bool(np.all(np.sqrt(self.group_squared(zd)) <= z_max))

    def containment(self, up1: dict, up2: dict, eps: float = 1e-15):
        p1 = self.secure_mix_parameters(up1, add_variance=True)
        p2 = self.secure_mix_parameters(up2, add_variance=True)
        w1 = p1["weights"]
        means1 = p1["measurement"]["pose"]["means"]
        vars1 = p1["measurement"]["pose"]["covariances"]
        pis1 = p1["measurement"]["state"]["pis"]
        w2 = p2["weights"]
        means2 = p2["measurement"]["pose"]["means"]
        vars2 = p2["measurement"]["pose"]["covariances"]
        pis2 = p2["measurement"]["state"]["pis"]

        chi = _chi2_ppf(self.cfg.z_quantile_joint, self.pose_dof)

        def agrees(i: int, j: int) -> bool:
            var1 = np.maximum(vars1[i], eps)
            var2 = np.maximum(vars2[j], eps)
            if not self._ellipsoids_intersect(
                means1[i],
                var1,
                means2[j],
                var2,
                chi,
                self._z_dim_sigma,
            ):
                return False
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
            np.ndarray of shape (N, FEATURE_DIM), filled block by block per
            :data:`Entity.LAYOUT`: the fitted state posterior over the live
            slots (summing to 1, unseen states at 0), then μ and log(σ) for
            position, orientation and the extra joint dims.

            Unlike :meth:`gnn_format`, which fills the log-std blocks with the
            ``BASE_LOGSTD`` sentinel, this writes the fitted log-stds, so both
            producers have to be scaled alike before they reach the encoder.
        """

        p = self.secure_mix_parameters(up)
        means = p["measurement"]["pose"]["means"]  # (N, n_columns)
        covariances = p["measurement"]["pose"]["covariances"]  # (N, n_columns)
        pis = p["measurement"]["state"]["pis"]  # (N, K)
        N = len(p["weights"])
        L = Entity.LAYOUT
        base = Entity.POS_DIM + (Entity.ROT_DIM if self.cfg.add_rotation else 0)
        D = means.shape[1] - base  # extra continuous dims beyond pos (+rot)
        if D > Entity.MAX_EXTRA_DIM:
            raise ValueError(
                f"fitted components have {D} extra dims but the feature layout "
                f"reserves Entity.MAX_EXTRA_DIM={Entity.MAX_EXTRA_DIM}"
            )
        self._check_state_width(self.cfg.n_states)

        feat = np.zeros((N, Entity.FEATURE_DIM), dtype=np.float32)

        n_logit = min(pis.shape[1], self.cfg.n_states)
        post = np.asarray(pis[:, :n_logit], dtype=np.float64)
        total = post.sum(axis=1, keepdims=True)
        post = np.divide(
            post, total, out=np.full_like(post, 1.0 / n_logit), where=total > 0.0
        )
        feat[:, L["state"].mean(n_logit)] = post

        # Position mean + logstd
        feat[:, L["pos"].mean()] = means[:, 0:3]
        feat[:, L["pos"].logstd()] = 0.5 * np.log(covariances[:, 0:3] + eps)

        # Axis-angle -> quaternion + rotation logstd (3 dof in tangent space)
        if self.cfg.add_rotation:
            quat = Quaternion.exp(means[:, 3:6])
            feat[:, L["rot"].mean()] = Quaternion.normalize(quat)
            feat[:, L["rot"].logstd()] = 0.5 * np.log(covariances[:, 3:6] + eps)
        else:
            feat[:, L["rot"].mean()] = Quaternion.identity()
            feat[:, L["rot"].logstd()] = self.BASE_LOGSTD

        # Extra continuous dims: mean + logstd
        if D > 0:
            feat[:, L["extra"].mean(D)] = means[:, base:]
            feat[:, L["extra"].logstd(D)] = 0.5 * np.log(covariances[:, base:] + eps)

        return feat, p["weights"]
