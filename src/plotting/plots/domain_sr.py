import matplotlib.pyplot as plt
import numpy as np
import src.plotting.helper as helper
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    data: dict[str, list] = {
        "domains": ["slider -> red", "red -> pink", "pink -> blue", "blue -> slider"],
        "gnn": [],
        "baseline": [],
    }

    for nt in ["gnn", "baseline"]:
        for domain in ["slider", "red", "pink", "blue"]:
            run_t = collection.get(
                nt=nt,
                mode="t",
                origin=domain,
                dest=domain,
                pe=0.0,
                pr=0.0,
            )
            run_e = collection.get(
                nt=nt,
                mode="e",
                origin=domain,
                dest=domain,
                pe=0.0,
                pr=0.0,
            )
            data[nt].append(
                run_e.stats["run_stats"]["max_sr"] - run_t.stats["run_stats"]["max_sr"]
            )

    x = np.arange(len(data["domains"]))
    width = 0.2

    fig, ax = plt.subplots(figsize=helper.FIG_SIZE_FLAT)

    gnn_values = np.array(data["gnn"])
    gnn_colors = np.where(
        gnn_values >= 0,
        helper.MAP_COLOR["baseline"]["main"],
        helper.MAP_COLOR["tree"]["secondary"],
    )

    ax.bar(
        x + width / 2,
        gnn_values,
        width,
        color=gnn_colors,
        label=f"{helper.MAP_LABEL['gnn']} Δ max SR",
    )

    baseline_values = np.array(data["baseline"])
    baseline_colors = np.where(baseline_values >= 0, "green", "red")

    ax.bar(
        x - width / 2,
        baseline_values,
        width,
        color=baseline_colors,
        label=f"{helper.MAP_LABEL['baseline']} Δ max SR",
    )

    ax.axhline(0, linewidth=1)
    # ax.set_ylim(-0.25, 0.25)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel("Δ max SR")
    ax.set_xlabel("Skill Set Transfer")
    ax.set_title("Change in max SR from Training to Evaluation for Each Domain")
    ax.legend()
    helper.save_plot("comparison_domain_sr.png")
