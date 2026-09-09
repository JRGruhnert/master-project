from functools import cached_property
from pathlib import Path
import warnings

from matplotlib import pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from stepmix import StepMix
import torch
from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.data.revolute import RevoluteEntity
from heca.misc import logger
from heca.utils.quaternion import Quaternion


class Condition:
    def __init__(
        self, label: str, data: dict[str, np.ndarray], entities: dict[str, Entity]
    ):
        self.data_raw = data
        self.entities = entities
        self.label = label

        self.models, self._bics = self._fit_model()

    def comp_features(
        self,
    ) -> dict[str, list[tuple[np.ndarray, float]]]:
        result: dict[str, list[tuple[np.ndarray, float]]] = {}
        for key in self.models.keys():
            up = self.models[key].get_parameters().copy()
            feats, weights = self.entities[key].comp_feature(up)
            result[key] = [(feats[i], float(weights[i])) for i in range(len(weights))]
        return result

    @cached_property
    def data_bounds(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, values in self.data_raw.items():
            values = np.asarray(values, dtype=np.float64)
            bounds[key] = (values.min(axis=0), values.max(axis=0))
        return bounds

    def test(self, elabel: str, x: DCScene) -> bool:
        up = self.models[elabel].get_parameters().copy()
        return self.entities[elabel].score_single(x[elabel].value, up)

    def sample(self, elabel: str) -> DCEntity:
        value = self.models[elabel].sample(1)[0]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = value.squeeze()
        entity = self.entities[elabel]
        value = entity.model_to_value(value)
        if (aa := self._default_aa(elabel, value)) is not None:
            value[Entity.POS_DIM : Entity.POS_DIM + Entity.ROT_DIM] = aa
        lo, hi = self.data_bounds[elabel]
        value = entity.sanitize_value(value, lo=lo, hi=hi)
        feat = entity.gnn_format(value)
        return DCEntity(value=value, feature=feat)

    def _default_aa(self, key: str, value: np.ndarray) -> np.ndarray | None:
        entity = self.entities[key]
        if entity.cfg.add_rotation:
            return None
        full = np.asarray(self.data_raw[key], dtype=np.float64)
        aas = full[:, Entity.POS_DIM : Entity.POS_DIM + Entity.ROT_DIM]
        if isinstance(entity, RevoluteEntity):
            extras = full[:, Entity.POS_DIM + Entity.ROT_DIM : -1]
            target = np.asarray(value, dtype=np.float64)[
                Entity.POS_DIM + Entity.ROT_DIM : -1
            ]
            i = int(np.argmin(np.linalg.norm(extras - target, axis=1)))
            return aas[i]
        quats = np.stack([Quaternion.exp(a) for a in aas])
        q = Quaternion.normalize(quats.mean(axis=0))
        if np.linalg.norm(q) < 0.5:
            return None
        return Quaternion.log_map(q)

    def _fit_model(self) -> tuple[dict[str, StepMix], dict[str, list[float]]]:
        models: dict[str, StepMix] = {}
        bics: dict[str, list[float]] = {}

        for key, values in self.data_raw.items():
            values = self.entities[key].model_value(values)
            best_model = None
            best_bic = np.inf
            bic_values: list[float] = []

            for k in range(1, self.entities[key].cfg.max_fit_components + 1):
                model = StepMix(
                    n_components=k,
                    measurement=self.entities[key].measurement,  # type: ignore
                    verbose=False,
                    progress_bar=0,
                )

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model.fit(values)
                if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                    logger.warning(
                        f"Did not converge: con={self.label} entity={key} n_comp={k}"
                    )
                bic = model.bic(values)
                bic_values.append(bic)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model

            assert best_model is not None
            models[key] = best_model
            bics[key] = bic_values

        return models, bics

    def plot(self, path: Path):
        ks = range(1, 10 + 1)
        plt.figure(figsize=(8, 5))
        for name, bic in self._bics.items():
            plt.plot(ks, bic, marker="o", label=f"{name}")
        plt.xlabel("Number of latent classes")
        plt.ylabel("BIC")
        plt.title("Model selection")
        plt.grid(True)
        plt.legend()
        plt.savefig(path / f"bic_{self.label}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def make_subgoal(self, other: "Condition") -> dict[str, np.ndarray] | None:
        values = {}
        for key in set(self.entities).intersection(set(other.entities)):
            up1 = self.models[key].get_parameters().copy()
            up2 = other.models[key].get_parameters().copy()
            if not self.entities[key].containment_score(up1, up2):
                return None
            value = self.entities[key].best_sample(up1, up2)
            if (aa := self._default_aa(key, value)) is not None:
                value[Entity.POS_DIM : Entity.POS_DIM + Entity.ROT_DIM] = aa
            values[key] = value
            logger.debug(f"{key}: value={value}")
        return values

    def scores(self, other: "Condition") -> dict[str, float]:
        result = {}
        for key in set(self.entities).intersection(set(other.entities)):
            up1 = self.models[key].get_parameters().copy()
            up2 = other.models[key].get_parameters().copy()
            score = self.entities[key].containment(up1, up2)
            result[key] = score
        return result
