from dataclasses import dataclass

import numpy as np
import torch

from heca.misc import logger
from heca.data.entity import Entity
from scripts.runners.plotter import HecaPlotter
from scripts.runners.plots.entity_3d import (
    Entity3DHelper,
    Entity3DHelperConfig,
    Entity3DHelperConfig,
    Entity3DMode,
    EntityPoint,
)


from heca.agents.agent import Agent


class HecaConditionPlotter(HecaPlotter):
    @dataclass
    class Config(HecaPlotter.Config):
        title: str = "Skill Conditions"
        subdir: str = "conditions"
        name: str = "conditions"
        entity3d: Entity3DHelperConfig = Entity3DHelperConfig(
            mode=Entity3DMode.DELTAS,
        )
        area: CalvinAreaConfig = CalvinAreaConfig()

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.entity3d = Entity3DHelper(cfg.entity3d)

    def reset(self):
        self.entity3d = Entity3DHelper(self.cfg.entity3d)

    def plot_content(self, agent: Agent):
        self.cfg.name = self.cfg.subdir + "_" + agent.cfg.query.label
        for entity in self.entities:
            label = entity.cfg.label
            self.entity3d.set_entity(
                entity=entity,
                current=self.make_dc_entity(label, agent.precons),
                goal=self.make_dc_entity(label, agent.postcons),
                different=True,  # Makes no sense otherwise
                solved=False,  # We dont support that currently
            )
        self.entity3d.show_entities()
        self.entity3d.show_areas(self.cfg.area)
        self.entity3d.finish()

    def make_dc_entity(
        self, label: str, conditions: dict[str, PropertyCondition]
    ) -> np.ndarray:

        if label in ["red", "blue", "pink"]:
            label = f"block_{label}"
        if label in ["slider", "drawer", "button", "led", "lightbulb"]:
            label = f"base__{label}"
        labels = [f"{label}_position", f"{label}_rotation", f"{label}_scalar"]
        skip = False
        logger.debug(f"Conditions keys: {conditions.keys()}")

        for l in labels:
            if l not in conditions:
                skip = True
                logger.debug(
                    f"Missing condition {l} for entity {label} in skill {self.cfg.name}"
                )
                break
        if not skip:
            return Entity.value_from_gt(
                pos=conditions[f"{label}_position"].value,
                rot=conditions[f"{label}_rotation"].value,
                ste=conditions[f"{label}_scalar"].value,
            )
        else:
            return np.empty(8)
