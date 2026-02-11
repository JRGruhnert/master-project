import matplotlib.pyplot as plt
import numpy as np
from src.plotting.run import RunData, RunDataCollection
from src.plotting.helper import *


def plot(collection: RunDataCollection):
    data: dict[str, list] = {
        "domains": ["slider -> red", "red -> pink", "pink -> blue", "blue -> slider"],
        NT_GNN: [],
        NT_MLP: [],
    }

    for nt in [NT_GNN, NT_MLP]:
        for domain in ["slider", "red", "pink", "blue"]:
            run_e = collection.get(
                nt=nt,
                mode=MODE_EVAL,
                origin=domain,
                dest=domain,
                pe=0.0,
                pr=0.0,
            )
            run_d = collection.get(
                nt=nt,
                mode=MODE_DOMAIN,
                origin=domain,
                dest=domain,
                pe=0.0,
                pr=0.0,
            )
            data[nt].append(
                run_d.stats["run_stats"]["max_sr"] - run_e.stats["run_stats"]["max_sr"]
            )

    x = np.arange(len(data["domains"]))
    width = 0.2

    fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

    gnn_values = np.array(data[NT_GNN])
    gnn_colors = np.where(
        gnn_values >= 0,
        MAP_COLOR[NT_GNN]["main"],
        MAP_COLOR[NT_TREE]["secondary"],
    )

    ax.bar(
        x + width / 2,
        gnn_values,
        width,
        color=gnn_colors,
        label=f"{MAP_LABEL[NT_GNN]}",
    )

    baseline_values = np.array(data[NT_MLP])
    baseline_colors = np.where(baseline_values >= 0, "green", "red")

    ax.bar(
        x - width / 2,
        baseline_values,
        width,
        color=baseline_colors,
        label=f"{MAP_LABEL[NT_MLP]}",
    )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel("Delta Difference")
    ax.set_xlabel("Skill Set Transfer")
    ax.set_title("Success Rate Difference between Domain and Evaluation")
    ax.legend()
    save_plot("comparison_domain_sr.png")
