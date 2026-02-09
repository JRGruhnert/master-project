import os
import matplotlib.pyplot as plt

# Sets a gloabal style. Every plot uses this still if this file is imported.
plt.style.use("seaborn-v0_8-white")
_SAVE_PATH = "plots"

# Global constants for plotting
FIG_SIZE = (10, 6)
FIG_SIZE_FLAT = (10, 6)
FIG_SIZE_HIGH = (10, 6)
SUBFIG_SIZE = (8, 5)

# Global easy access constants for mapping

MAP_COLOR = {
    "gnn": {
        "main": "",
        "secondary": "",
    },
    "mlp": {
        "main": "",
        "secondary": "",
    },
    "tree": {
        "main": "",
        "secondary": "",
    },
}
MAP_LABEL = {"gnn": "GNN", "mlp": "MLP", "tree": "Tree"}

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


def plot_threshold_lines():
    # Add threshold lines
    plt.axhline(y=0.1, color="black", linestyle="--", alpha=0.25, label="10%")
    plt.axhline(y=0.2, color="black", linestyle="--", alpha=0.25, label="20%")
    plt.axhline(y=0.3, color="black", linestyle="--", alpha=0.25, label="30%")
    plt.axhline(y=0.4, color="black", linestyle="--", alpha=0.25, label="40%")
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.25, label="50%")
    plt.axhline(y=0.6, color="black", linestyle="--", alpha=0.25, label="60%")
    plt.axhline(y=0.7, color="black", linestyle="--", alpha=0.25, label="70%")
    plt.axhline(y=0.8, color="black", linestyle="--", alpha=0.25, label="80%")
    plt.axhline(y=0.9, color="black", linestyle="--", alpha=0.25, label="90%")
    plt.axhline(y=0.95, color="red", linestyle="--", alpha=0.25, label="95%")


def save_plot(filename: str, subdir: str = ""):
    """Save plot to specified path"""
    save_dir = os.path.join(_SAVE_PATH, subdir)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot: {plot_path}")
