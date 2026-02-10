import matplotlib.pyplot as plt
import numpy as np
from src.plotting.run import RunData, RunDataCollection
from src.plotting.helper import (
    MAP_LABEL,
    MAP_COLOR,
    LIST_DOMAIN,
    FIG_SIZE,
    NT_MLP,
    NT_GNN,
    NT_TREE,
    save_plot,
    set_y_ticks,
)
import src.plotting.helper as helper


def plot(collection: RunDataCollection):
    # Example data
    domains = ["Domain A", "Domain B", "Domain C", "Domain D"]

    epochs_gnn = [120, 95, 140, 110]
    epochs_mlp = [180, 160, 210, 175]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots()

    ax.bar(x - width / 2, epochs_gnn, width, label=MAP_LABEL[NT_GNN])
    ax.bar(x + width / 2, epochs_mlp, width, label=MAP_LABEL[NT_MLP])

    ax.set_xlabel("Domain Sets")
    ax.set_ylabel("Epochs / Count until X%")
    ax.set_title("All Comparison Time Plot")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    set_y_ticks(ax)
    save_plot("comparison_all_time.png")
