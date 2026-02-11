import matplotlib.pyplot as plt
from src.plotting.helper import *
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    x_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    data: dict[str, dict[str, list[float]]] = {
        "pe": {NT_GNN: [], NT_MLP: []},
        "pr": {NT_GNN: [], NT_MLP: []},
    }
    for p in ["pe", "pr"]:
        for nt in [NT_GNN, NT_MLP]:
            for value in x_values:
                # Collect data for plotting
                if value == 1.0:
                    if p == "pe":
                        data[p][nt].append(0.0)
                    else:
                        run = collection.get(
                            nt=NT_MLP,
                            mode=MODE_TRAIN,
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
                            mode=MODE_TRAIN,
                            origin="slider",
                            dest="slider",
                            pe=value,
                            pr=0.0,
                        )
                    else:
                        run = collection.get(
                            nt=nt,
                            mode=MODE_TRAIN,
                            origin="slider",
                            dest="slider",
                            pe=0.0,
                            pr=value,
                        )
                    data[p][nt].append(run.stats["run_stats"]["max_sr"])

    # Plot each tag with all networks
    for p_tag in data.keys():
        plt.figure(figsize=FIG_SIZE)
        for network in data[p_tag].keys():
            plt.plot(
                x_values,
                data[p_tag][network],
                color=MAP_COLOR[network]["main"],
                label=network.upper(),
            )

        plt.xlabel(MAP_LABEL[p_tag])
        plt.ylabel(LABEL_SR)
        # plt.title(f"Network Comparison: {p_tag}")
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.0)
        set_y_ticks()
        save_plot(f"comparison_{p_tag}.png")
