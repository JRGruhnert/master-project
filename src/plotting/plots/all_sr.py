import matplotlib.pyplot as plt
import numpy as np
from run import RunData, RunDataCollection
from helper import (
    MAP_LABEL,
    MAP_COLOR,
    LIST_DOMAIN,
    FIG_SIZE,
    NT_MLP,
    NT_GNN,
    NT_TREE,
    MODE_TRAIN,
    save_plot,
)


def plot(collection: RunDataCollection):
    """Compare max_sr vs parameter across all networks for each tag
    Args:
        data: dict[tag][network] = p_tag, network -> list of max_sr

    """
    sr: dict[str, list[float]] = {NT_GNN: [], NT_MLP: [], NT_TREE: []}

    for set in LIST_DOMAIN:
        for nt in [NT_GNN, NT_MLP, NT_TREE]:
            run = collection.get(
                nt=nt,
                mode=MODE_TRAIN,
                origin=set,
                dest=set,
                pe=0.0,
                pr=0.0,
            )
            sr[nt].append(run.stats["run_stats"]["max_sr"])

    x = np.arange(len(LIST_DOMAIN))
    width = 0.25

    fig, ax = plt.subplots()

    ax.bar(x - width, sr[NT_MLP], width, label=MAP_LABEL[NT_MLP])
    ax.bar(x, sr[NT_GNN], width, label=MAP_LABEL[NT_GNN])
    ax.bar(x + width, sr[NT_TREE], width, label=MAP_LABEL[NT_TREE])

    ax.set_xlabel("Skill Sets")
    ax.set_ylabel("Maximal Success Rate")
    ax.set_title("All Comparison SR Plot")
    ax.set_xticks(x)
    ax.set_xticklabels(LIST_DOMAIN)
    ax.set_ylim(0, 1.0)

    ax.legend()

    plt.tight_layout()
    plt.show()
    save_plot("comparison_all_sr.png")
