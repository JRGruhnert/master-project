from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from heca.misc import logger
from heca.data.entity import Entity
from heca.misc.base import Configurable
from heca.properties.property import PropertyV1
from heca.agents.agent import Agent

# Sets a gloabal style. Every plot uses this still if this file is imported.
plt.style.use("seaborn-v0_8")

plt.rcParams.update(
    {
        "font.size": 20,
        "font.weight": "bold",
    }
)
FONT_SIZE = 20
FONT_SIZE_MEDIUM = 18
FONT_SIZE_SMALLER = 16

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "font.size": FONT_SIZE_MEDIUM,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE_MEDIUM,
        "xtick.labelsize": FONT_SIZE_SMALLER,
        "ytick.labelsize": FONT_SIZE_SMALLER,
        "legend.fontsize": FONT_SIZE_SMALLER,
        "figure.titlesize": FONT_SIZE,
    }
)
# Global constants for plotting
FIG_SIZE = (10, 6)
FIG_SIZE_FLAT = (12, 6)
FIG_SIZE_HIGH = (8, 6)
HATCH_PATTERN = "xxxxx"

LABEL_EPOCH = "Epoch"
LABEL_REWARD = "Mean Reward"
LABEL_SR = "Success Rate"
LABEL_LENGTH = "Mean Episode Length"
LABEL_SKILLSET = "Skill Set"
# Global easy access constants for mapping

MAP_COLOR = {
    "gnn": {
        "main": "#1557c0",
        "secondary": "#0f3375",
        "tertiary": "#00a6fb",
    },
    "baseline": {
        "main": "#1a7431",
        "secondary": "#10451d",
        "tertiary": "#4ad66d",
    },
    "tree": {
        "main": "#9f040e",
        "secondary": "#a51c30",
    },
}
MAP_LABEL = {
    "gnn": "GNN",
    "baseline": "MLP",
    "tree": "Tree",
    "pe": "Percentage of skipped actions",
    "pr": "Percentage of random actions",
}

# Lists for easy access
LIST_DOMAIN = ["slider", "red", "pink", "blue", "sr", "srp", "srpb"]
# Lists for easy access
LIST_DOMAIN_SMALL = ["slider", "blue", "red", "pink"]

# Network Types
NT_GNN = "gnn"
NT_MLP = "baseline"
NT_TREE = "tree"


# Mode Types
MODE_TRAIN = "t"
MODE_EVAL = "e"
MODE_DOMAIN = "d"
MODE_RETRAIN = "r"
MODE_RETRAIN_EVAL = "re"
MODE_RETRAIN_DOMAIN = "rd"


LEGEND_WITHOUT_TREE_AND_EVAL = [
    mpatches.Patch(facecolor=MAP_COLOR[NT_MLP]["main"], label=MAP_LABEL[NT_MLP]),
    mpatches.Patch(facecolor=MAP_COLOR[NT_GNN]["main"], label=MAP_LABEL[NT_GNN]),
]
LEGEND_WITHOUT_TREE = [
    mpatches.Patch(facecolor=MAP_COLOR[NT_MLP]["main"], label=MAP_LABEL[NT_MLP]),
    mpatches.Patch(facecolor=MAP_COLOR[NT_GNN]["main"], label=MAP_LABEL[NT_GNN]),
    mpatches.Patch(
        facecolor="white",
        hatch=HATCH_PATTERN,
        label="Evaluation",
        edgecolor="black",
        linewidth=1.0,
    ),
]

LEGEND_RETRAIN = [
    mpatches.Patch(
        facecolor=MAP_COLOR[NT_MLP]["main"], label=MAP_LABEL[NT_MLP] + " scratch"
    ),
    mpatches.Patch(
        facecolor=MAP_COLOR[NT_GNN]["main"], label=MAP_LABEL[NT_GNN] + " scratch"
    ),
    mpatches.Patch(
        facecolor=MAP_COLOR[NT_MLP]["secondary"], label=MAP_LABEL[NT_MLP] + " retrain"
    ),
    mpatches.Patch(
        facecolor=MAP_COLOR[NT_GNN]["secondary"], label=MAP_LABEL[NT_GNN] + " retrain"
    ),
    mpatches.Patch(
        facecolor="white",
        hatch=HATCH_PATTERN,
        label="Evaluation",
        edgecolor="black",
        linewidth=1.0,
    ),
    mpatches.Patch(
        facecolor="white",
        hatch=HATCH_PATTERN,
        label="Zero-Shot",
        edgecolor=MAP_COLOR[NT_TREE]["main"],
        linewidth=1.0,
    ),
]

LEGEND_WITH_TREE = [
    mpatches.Patch(facecolor=MAP_COLOR[NT_MLP]["main"], label=MAP_LABEL[NT_MLP]),
    mpatches.Patch(facecolor=MAP_COLOR[NT_GNN]["main"], label=MAP_LABEL[NT_GNN]),
    mpatches.Patch(facecolor=MAP_COLOR[NT_TREE]["main"], label=MAP_LABEL[NT_TREE]),
    mpatches.Patch(
        facecolor="white",
        hatch=HATCH_PATTERN,
        label="Evaluation",
        edgecolor="black",
        linewidth=1.0,
    ),
]


def smooth_data(data, window_size=5):
    kernel = np.ones(window_size) / window_size
    pad = window_size // 2
    data = np.pad(data, pad, mode="edge")  # or "reflect"
    return np.convolve(data, kernel, mode="valid")


def set_y_ticks(ax=None, step=0.1, ymin=0.0, ymax=1.0):
    """Set y-axis ticks at regular intervals"""
    if ax is None:
        ax = plt.gca()

    ax.set_yticks(np.arange(ymin, ymax + step, step))
    ax.set_ylim(ymin, ymax)


def get_color_map(x: float, special_color="gray", cmap_name="tab10"):
    cmap = plt.get_cmap(cmap_name)

    def color_fn(i):
        if i == -1:
            return special_color
        else:
            return cmap((i - 1) % x)

    return color_fn


@dataclass
class StyleConfig:
    grey: str = "grey"
    black: str = "black"
    white: str = "white"
    solved: str = "tab:green"
    unsolved: str = "tab:red"
    true: str = "tab:yellow"
    false: str = "black"
    entity: dict[str, str] = field(
        default_factory=lambda: {
            "ee": "black",
            "block_red": "tab:red",
            "block_blue": "tab:blue",
            "block_pink": "tab:pink",
            "slide": "black",
            "drawer": "black",
            "button": "black",
            "led": "black",
            "lightbulb": "black",
        }
    )


class HecaPlotter(Configurable):
    @dataclass
    class Config(Configurable.Config):
        title: str
        name: str
        subdir: str
        rootdir: str = "plots"
        style: StyleConfig = StyleConfig()
        dry_run: bool = False
        agents: list[Agent.Config] = field(default_factory=list)
        # plots: list[Plot.Config]

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def make_content(self, plot, *args, **kwargs):
        raise NotImplementedError()

    def plot(self):
        pass
