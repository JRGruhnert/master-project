import matplotlib.pyplot as plt
import numpy as np
import src.plotting.helper as helper
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    data: dict[str, list] = {
        "domains": ["slider -> red", "red -> pink", "pink -> blue", "blue -> slider"],
        "gnn": [],
        "mlp": [],
    }

    for nt in ["gnn", "mlp"]:
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
    width = 0.35

    fig, ax = plt.subplots()

    for i, model in enumerate(["gnn", "baseline"]):
        values = np.array(data[model])
        colors = np.where(values >= 0, "green", "red")

        ax.bar(
            x + i * width,
            values,
            width,
            color=colors,
            label=model,
        )

    ax.axhline(0, linewidth=1)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel("Δ max SR")
    ax.legend()
    helper.save_plot("comparison_domain_sr.png")
