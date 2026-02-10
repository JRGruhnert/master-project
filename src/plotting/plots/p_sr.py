import matplotlib.pyplot as plt
import src.plotting.helper as helper
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    x_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data: dict[str, dict[str, list[float]]] = {
        "pe": {"gnn": [], "baseline": []},
        "pr": {"gnn": [], "baseline": []},
    }
    for p in ["pe", "pr"]:
        for nt in ["gnn", "baseline"]:
            for value in x_values:
                # Collect data for plotting
                if value == 1.0:
                    if p == "pe":
                        data[p][nt].append(0.0)
                    else:
                        run = collection.get(
                            nt="baseline",
                            mode="t",
                            origin="slider",
                            dest="slider",
                            pe=0.0,
                            pr=value,
                        )
                        data[p][nt].append(run.stats["run_stats"]["max_sr"])
                else:
                    if p == "pe":
                        run = collection.get(
                            nt=nt,
                            mode="t",
                            origin="slider",
                            dest="slider",
                            pe=value,
                            pr=0.0,
                        )
                    else:
                        run = collection.get(
                            nt=nt,
                            mode="t",
                            origin="slider",
                            dest="slider",
                            pe=0.0,
                            pr=value,
                        )
                    data[p][nt].append(run.stats["run_stats"]["max_sr"])

    # Plot each tag with all networks
    for p_tag in data.keys():
        plt.figure(figsize=helper.FIG_SIZE)
        for network in data[p_tag].keys():
            plt.plot(
                x_values,
                data[p_tag][network],
                markersize=10,
                linewidth=2.5,
                alpha=0.7,
                color=helper.MAP_COLOR[network]["main"],
                markeredgewidth=1.5,
                label=network.upper(),
            )

        plt.xlabel(
            f"Percentage of Alternation",
            fontsize=13,
            fontweight="bold",
        )
        plt.ylabel("Maximum Success Rate", fontsize=13, fontweight="bold")
        plt.title(
            f"Network Comparison: {p_tag}",
            fontsize=15,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        plt.legend(fontsize=11, loc="best")
        helper.set_y_ticks()
        helper.save_plot(f"comparison_{p_tag}.png")
