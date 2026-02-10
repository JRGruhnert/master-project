import matplotlib.pyplot as plt
import numpy as np
import src.plotting.helper as helper
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    data: dict[str, list] = {
        "domains": ["sr", "srp", "srpb"],
        "gnn_scratch": [],
        "baseline_scratch": [],
        "gnn_retrain": [],
        "baseline_retrain": [],
        "gnn_improvement": [],
        "baseline_improvement": [],
    }
    for nt in ["gnn", "baseline"]:
        for domain in data["domains"]:
            run_t = collection.get(
                nt=nt, mode="t", origin="srpb", dest=domain, pe=0.0, pr=0.0
            )
            data[nt + "_scratch"].append(run_t.stats["run_stats"]["max_sr"])

            if domain != "sr":
                run_r = collection.get(
                    nt=nt, mode="r", origin="srpb", dest=domain, pe=0.0, pr=0.0
                )
                data[nt + "_retrain"].append(run_r.stats["batch_stats"][0]["max_sr"])
                data[nt + "_improvement"].append(
                    run_r.stats["run_stats"]["max_sr"]
                    - run_r.stats["batch_stats"][0]["max_sr"]
                )
            else:
                data[nt + "_retrain"].append(0.0)
                data[nt + "_improvement"].append(0.0)

    n_domains = len(data["domains"])
    n_bars = 4  # 2 GNN + 2 Baseline
    width = 0.1
    group_gap = 0.04  # Gap between GNN and Baseline within domain

    # Total width of one domain group
    group_width = n_bars * width + group_gap

    # X positions for domains (spread out enough to not overlap)
    x = np.arange(n_domains) * (group_width + 0.08)

    fig, ax = plt.subplots(figsize=helper.FIG_SIZE_FLAT)

    # GNN bars (left side of each domain)
    ax.bar(
        x - width * 1.5 - group_gap / 2,
        data["gnn_scratch"],
        width,
        color=helper.MAP_COLOR["gnn"]["main"],
        label=f"{helper.MAP_LABEL['gnn']} scratch",
    )
    ax.bar(
        x - width * 0.5 - group_gap / 2,
        data["gnn_retrain"],
        width,
        color=helper.MAP_COLOR["gnn"]["secondary"],
        label=f"{helper.MAP_LABEL['gnn']} retrain (start)",
    )
    ax.bar(
        x - width * 0.5 - group_gap / 2,
        data["gnn_improvement"],
        width,
        bottom=data["gnn_retrain"],
        color=helper.MAP_COLOR["gnn"]["tertiary"],
        label=f"{helper.MAP_LABEL['gnn']} retrain (improvement)",
    )

    # Baseline bars (right side of each domain)
    ax.bar(
        x + width * 0.5 + group_gap / 2,
        data["baseline_scratch"],
        width,
        color=helper.MAP_COLOR["baseline"]["main"],
        label=f"{helper.MAP_LABEL['baseline']} scratch",
    )
    ax.bar(
        x + width * 1.5 + group_gap / 2,
        data["baseline_retrain"],
        width,
        color=helper.MAP_COLOR["baseline"]["secondary"],
        label=f"{helper.MAP_LABEL['baseline']} retrain (start)",
    )
    ax.bar(
        x + width * 1.5 + group_gap / 2,
        data["baseline_improvement"],
        width,
        bottom=data["baseline_retrain"],
        color=helper.MAP_COLOR["baseline"]["tertiary"],
        label=f"{helper.MAP_LABEL['baseline']} retrain (improvement)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel("Max Success Rate")
    ax.legend()
    helper.set_y_ticks(ax)
    helper.save_plot("comparison_retrain_sr.png")
