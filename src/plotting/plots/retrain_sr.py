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
                data[nt + "_retrain"].append(run_r.stats["run_stats"]["max_sr"])
                data[nt + "_improvement"].append(
                    run_r.stats["run_stats"]["max_sr"]
                    - run_t.stats["run_stats"]["max_sr"]
                )
            else:
                data[nt + "_retrain"].append(0.0)
                data[nt + "_improvement"].append(0.0)

    x = np.arange(len(data["domains"]))
    width = 0.35

    fig, ax = plt.subplots()

    # scratch bar
    ax.bar(
        x - width / 2,
        data[f"{nt}_scratch"],
        width,
        label="scratch",
    )

    # retrain base (starting SR)
    ax.bar(
        x + width / 2,
        data[f"{nt}_retrain"],
        width,
        color="lightgray",
        label="retrain (start)",
    )

    # retrain improvement
    ax.bar(
        x + width / 2,
        data[f"{nt}_improvement"],
        width,
        bottom=data[f"{nt}_retrain"],
        color="green",
        label="retrain (gain)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(data["domains"])
    ax.set_ylabel("max SR")
    ax.legend()
    helper.save_plot("comparison_retrain_sr.png")
