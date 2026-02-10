import matplotlib.pyplot as plt
import numpy as np
from src.plotting.run import RunData, RunDataCollection
from src.plotting.helper import (
    EVAL,
    FIG_SIZE_FLAT,
    LABEL_MAX_SR,
    LABEL_SKILLSET,
    MAP_LABEL,
    MAP_COLOR,
    LIST_DOMAIN,
    FIG_SIZE,
    MODE_EVAL,
    NT_MLP,
    NT_GNN,
    NT_TREE,
    MODE_TRAIN,
    TRAIN,
    save_plot,
    set_y_ticks,
)


def plot(collection: RunDataCollection):
    """Compare max_sr vs parameter across all networks for each tag
    Args:
        data: dict[tag][network] = p_tag, network -> list of max_sr

    """
    sr: dict[str, dict[str, list[float]]] = {
        NT_GNN: {EVAL: [], TRAIN: []},
        NT_MLP: {EVAL: [], TRAIN: []},
        NT_TREE: {EVAL: [], TRAIN: []},
    }

    for set in LIST_DOMAIN:
        for nt in [NT_GNN, NT_MLP, NT_TREE]:
            if nt == NT_TREE:
                try:
                    run = collection.get(
                        nt=nt,
                        mode=MODE_EVAL,
                        origin=set,
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                    )
                except ValueError:
                    run = None
                if run is not None:
                    sr[nt][TRAIN].append(run.stats["run_stats"]["max_sr"])
                else:
                    sr[nt][TRAIN].append(0.0)
            else:
                if set in ["sr", "srp", "srpb"]:
                    run_t = collection.get(
                        nt=nt,
                        mode=MODE_TRAIN,
                        origin="srpb",
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                    )
                    run_e = collection.get(
                        nt=nt,
                        mode=MODE_EVAL,
                        origin="srpb",
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                    )

                else:
                    run_t = collection.get(
                        nt=nt,
                        mode=MODE_TRAIN,
                        origin=set,
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                    )
                    run_e = collection.get(
                        nt=nt,
                        mode=MODE_EVAL,
                        origin=set,
                        dest=set,
                        pe=0.0,
                        pr=0.0,
                    )
                sr[nt][TRAIN].append(
                    run_t.stats["run_stats"]["max_sr"]
                    - run_e.stats["run_stats"]["max_sr"]
                )

                sr[nt][EVAL].append(run_e.stats["run_stats"]["max_sr"])

    x = np.arange(len(LIST_DOMAIN))
    width = 0.20

    fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

    ax.bar(
        x - width,
        sr[NT_MLP][EVAL],
        width,
        hatch="//",
        label=MAP_LABEL[NT_MLP],
        color=MAP_COLOR[NT_MLP]["main"],
    )
    ax.bar(
        x,
        sr[NT_GNN][EVAL],
        width,
        hatch="//",
        label=MAP_LABEL[NT_GNN],
        color=MAP_COLOR[NT_GNN]["main"],
    )
    ax.bar(
        x - width,
        sr[NT_MLP][TRAIN],
        width,
        bottom=sr[NT_MLP][EVAL],
        label=MAP_LABEL[NT_MLP],
        color=MAP_COLOR[NT_MLP]["main"],
    )
    ax.bar(
        x,
        sr[NT_GNN][TRAIN],
        width,
        bottom=sr[NT_GNN][EVAL],
        label=MAP_LABEL[NT_GNN],
        color=MAP_COLOR[NT_GNN]["main"],
    )
    ax.bar(
        x + width,
        sr[NT_TREE][TRAIN],
        width,
        label=MAP_LABEL[NT_TREE],
        color=MAP_COLOR[NT_TREE]["main"],
    )

    ax.set_xlabel(LABEL_SKILLSET)
    ax.set_ylabel(LABEL_MAX_SR)
    ax.set_xticks(x)
    set_y_ticks(ax)
    ax.set_xticklabels(LIST_DOMAIN)
    ax.set_title("Comparison of Max Success Rates across Skill Sets")
    ax.legend()
    save_plot("comparison_all_sr.png")
