import matplotlib.pyplot as plt
import numpy as np
from src.plotting.run import RunData, RunDataCollection
from src.plotting.helper import (
    FIG_SIZE_FLAT,
    MAP_LABEL,
    MAP_COLOR,
    LIST_DOMAIN,
    FIG_SIZE,
    MODE_EVAL,
    NT_MLP,
    NT_GNN,
    NT_TREE,
    MODE_TRAIN,
    save_plot,
    set_y_ticks,
)


def plot(collection: RunDataCollection):
    """Compare max_sr vs parameter across all networks for each tag
    Args:
        data: dict[tag][network] = p_tag, network -> list of max_sr

    """
    sr: dict[str, list[float]] = {NT_GNN: [], NT_MLP: []}

    for set in LIST_DOMAIN:
        for nt in [NT_GNN, NT_MLP]:
            if set in ["sr", "srp", "srpb"]:
                run = collection.get(
                    nt=nt,
                    mode=MODE_TRAIN,
                    origin="srpb",
                    dest=set,
                    pe=0.0,
                    pr=0.0,
                )
            else:
                run = collection.get(
                    nt=nt,
                    mode=MODE_TRAIN,
                    origin=set,
                    dest=set,
                    pe=0.0,
                    pr=0.0,
                )
            sr[nt].append(run.stats["run_stats"]["sr_until_80"])

    x = np.arange(len(LIST_DOMAIN))
    width = 0.20

    fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

    ax.bar(
        x - width / 2,
        sr[NT_MLP],
        width,
        label=MAP_LABEL[NT_MLP],
        color=MAP_COLOR[NT_MLP]["main"],
    )
    ax.bar(
        x + width / 2,
        sr[NT_GNN],
        width,
        label=MAP_LABEL[NT_GNN],
        color=MAP_COLOR[NT_GNN]["main"],
    )

    ax.set_xlabel("Skill Sets")
    ax.set_ylabel("Epochs")
    ax.set_title("All Comparison Time Plot")
    ax.set_xticks(x)
    ax.set_xticklabels(LIST_DOMAIN)
    ax.legend()
    set_y_ticks(ax)
    save_plot("comparison_all_time.png")
