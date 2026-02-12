import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

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
_SAVE_PATH = "plots"
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


def save_plot(filename: str, subdir: str = ""):
    """Save plot to specified path"""
    save_dir = os.path.join(_SAVE_PATH, subdir)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot: {plot_path}")
