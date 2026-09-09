import matplotlib.pyplot as plt
import numpy as np
from heca.misc.run_data import TrainingRunData, RunDataCollection
from heca.plots.helper.helper import *


def plot(collection: RunDataCollection):
    data: dict[str, list] = {
        "domains": [
            #    "blue -> slider",
            #    "slider -> red",
            #    "red -> pink",
            #    "pink -> blue",
            "blue -> slider",
            "slider -> blue",
            "blue -> pink",
            "pink -> blue",
            "blue -> red",
            "red -> blue",
        ],
        NT_GNN: [],
        NT_MLP: [],
    }

    for nt in [NT_GNN, NT_MLP]:
        for domain in [
            #    ["blue", "slider"],
            #    ["slider", "red"],
            #    ["red", "pink"],
            #    ["pink", "blue"],
            ["blue", "slider"],
            ["slider", "red"],
            ["red", "pink"],
            ["pink", "blue"],
            ["red", "green"],
            ["pink", "yellow"],
        ]:
            run_t = collection.get(
                nt=nt,
                mode=MODE_EVAL,
                origin=domain[0],
                dest=domain[0],
                pe=0.0,
                pr=0.0,
            )
            run_d = collection.get(
                nt=nt,
                mode=MODE_DOMAIN,
                origin=domain[1],
                dest=domain[1],
                pe=0.0,
                pr=0.0,
            )
            data[nt].append(
                run_d.stats["run_stats"]["max_sr"] - run_t.stats["run_stats"]["max_sr"]
            )

    x = np.arange(len(data["domains"]))
    width = 0.2

    fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

    gnn_values = np.array(data[NT_GNN])

    ax.bar(
        x + width / 2,
        gnn_values,
        width,
        color=MAP_COLOR[NT_GNN]["main"],
        label=f"{MAP_LABEL[NT_GNN]}",
    )

    baseline_values = np.array(data[NT_MLP])

    ax.bar(
        x - width / 2,
        baseline_values,
        width,
        color=MAP_COLOR[NT_MLP]["main"],
        label=f"{MAP_LABEL[NT_MLP]}",
    )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel("Delta Difference")
    ax.set_xlabel("Skill Set Transfer")
    ax.set_title("Zero-Shot Evaluation between Skill Sets")
    ax.legend()
    save_plot("comparison_domain_sr.png")
