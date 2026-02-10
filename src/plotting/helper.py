import os
import matplotlib.pyplot as plt
import numpy as np

# Sets a gloabal style. Every plot uses this still if this file is imported.
plt.style.use("seaborn-v0_8")
_SAVE_PATH = "plots"

# Global constants for plotting
FIG_SIZE = (10, 6)
FIG_SIZE_FLAT = (12, 6)
FIG_SIZE_HIGH = (10, 6)
SUBFIG_SIZE = (8, 5)

# Global easy access constants for mapping

MAP_COLOR = {
    "gnn": {
        "main": "#003554",
        "secondary": "#006494",
        "tertiary": "#00a6fb",
    },
    "baseline": {
        "main": "#10451d",
        "secondary": "#1a7431",
        "tertiary": "#4ad66d",
    },
    "tree": {
        "main": "#580c1f",
        "secondary": "#a51c30",
    },
}
MAP_LABEL = {"gnn": "GNN", "baseline": "MLP", "tree": "Tree"}

# Lists for easy access
LIST_DOMAIN = ["slider", "blue", "red", "pink", "sr", "srp", "srpb"]

# Network Types
NT_GNN = "gnn"
NT_MLP = "baseline"
NT_TREE = "tree"


# Mode Types
MODE_TRAIN = "t"
MODE_EVAL = "e"
MODE_RETRAIN = "r"


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
