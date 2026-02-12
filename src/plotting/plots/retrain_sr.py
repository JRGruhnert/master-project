import matplotlib.pyplot as plt
import numpy as np
from src.plotting.helper import *
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    data: dict[str, list] = {
        "domains": ["sr", "srp", "srpb"],
        "gnn_train": [],
        "gnn_train_eval": [],
        "baseline_train": [],
        "baseline_train_eval": [],
        "gnn_retrain": [],
        "gnn_retrain_eval": [],
        "baseline_retrain": [],
        "baseline_retrain_eval": [],
    }
    for nt in [NT_GNN, NT_MLP]:
        for domain in data["domains"]:
            run_t = collection.get(
                nt=nt, mode=MODE_TRAIN, origin="srpb", dest=domain, pe=0.0, pr=0.0
            )
            run_e = collection.get(
                nt=nt, mode=MODE_EVAL, origin="srpb", dest=domain, pe=0.0, pr=0.0
            )
            data[nt + "_train"].append(
                run_t.stats["run_stats"]["max_sr"] - run_e.stats["run_stats"]["max_sr"]
            )
            data[nt + "_train_eval"].append(run_e.stats["run_stats"]["max_sr"])

            if domain != "sr":
                run_r = collection.get(
                    nt=nt,
                    mode=MODE_RETRAIN,
                    origin="srpb",
                    dest=domain,
                    pe=0.0,
                    pr=0.0,
                )
                run_r_e = collection.get(
                    nt=nt,
                    mode=MODE_RETRAIN_EVAL,
                    origin="srpb",
                    dest=domain,
                    pe=0.0,
                    pr=0.0,
                )
                data[nt + "_retrain"].append(
                    run_r.stats["batch_stats"]["max_sr"]
                    - run_r_e.stats["run_stats"]["max_sr"]
                )
                data[nt + "_retrain_eval"].append(run_r_e.stats["run_stats"]["max_sr"])
            else:
                data[nt + "_retrain"].append(0.0)
                data[nt + "_retrain_eval"].append(0.0)

    n_domains = len(data["domains"])
    n_bars = 4  # 2 GNN + 2 Baseline
    width = 0.1
    group_gap = 0.04  # Gap between GNN and Baseline within domain

    # Total width of one domain group
    group_width = n_bars * width + group_gap

    # X positions for domains (spread out enough to not overlap)
    x = np.arange(n_domains) * (group_width + 0.08)

    fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

    # GNN bars (left side of each domain)
    ax.bar(
        x - width * 1.5 - group_gap / 2,
        data["gnn_train_eval"],
        width,
        edgecolor="black",
        linewidth=1.0,
        hatch=HATCH_PATTERN,
        color=MAP_COLOR[NT_GNN]["main"],
        label=f"{MAP_LABEL[NT_GNN]} train eval",
    )
    ax.bar(
        x - width * 1.5 - group_gap / 2,
        data["gnn_train"],
        width,
        bottom=data["gnn_train_eval"],
        color=MAP_COLOR[NT_GNN]["main"],
        label=f"{MAP_LABEL[NT_GNN]} train",
    )
    ax.bar(
        x - width * 0.5 - group_gap / 2,
        data["gnn_retrain_eval"],
        width,
        edgecolor="black",
        linewidth=1.0,
        hatch=HATCH_PATTERN,
        color=MAP_COLOR[NT_GNN]["secondary"],
        label=f"{MAP_LABEL[NT_GNN]} retrain eval",
    )
    ax.bar(
        x - width * 0.5 - group_gap / 2,
        data["gnn_retrain"],
        width,
        bottom=data["gnn_retrain_eval"],
        color=MAP_COLOR[NT_GNN]["secondary"],
        label=f"{MAP_LABEL[NT_GNN]} retrain eval",
    )

    # Baseline bars (right side of each domain)
    ax.bar(
        x + width * 0.5 + group_gap / 2,
        data["baseline_train_eval"],
        width,
        edgecolor="black",
        linewidth=1.0,
        hatch=HATCH_PATTERN,
        color=MAP_COLOR[NT_MLP]["main"],
        label=f"{MAP_LABEL[NT_MLP]} train eval",
    )
    ax.bar(
        x + width * 0.5 + group_gap / 2,
        data["baseline_train"],
        width,
        bottom=data["baseline_train_eval"],
        color=MAP_COLOR[NT_MLP]["main"],
        label=f"{MAP_LABEL[NT_MLP]} train",
    )
    ax.bar(
        x + width * 1.5 + group_gap / 2,
        data["baseline_retrain_eval"],
        width,
        edgecolor="black",
        linewidth=1.0,
        hatch=HATCH_PATTERN,
        color=MAP_COLOR[NT_MLP]["secondary"],
        label=f"{MAP_LABEL[NT_MLP]} retrain eval",
    )
    ax.bar(
        x + width * 1.5 + group_gap / 2,
        data["baseline_retrain"],
        width,
        bottom=data["baseline_retrain_eval"],
        color=MAP_COLOR[NT_MLP]["secondary"],
        label=f"{MAP_LABEL[NT_MLP]} retrain",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel(LABEL_SR)
    ax.set_title("Comparison of Max Success Rates across Skill Sets")
    ax.legend()
    set_y_ticks(ax)
    save_plot("comparison_retrain_sr.png")
