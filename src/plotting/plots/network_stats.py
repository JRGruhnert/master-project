import matplotlib.pyplot as plt
import numpy as np
import src.plotting.helper as helper


def plot():
    # Data from your measurements
    domains = ["slider", "red", "pink", "blue", "sr", "srp", "srpb"]

    gnn_flops = [125840, 125936, 125936, 125936, 161840, 197840, 233840]
    gnn_params = [12754, 13410, 13410, 13410, 13410, 13410, 13410]

    baseline_flops = [1021568, 1021664, 1021664, 1021664, 1502768, 2076512, 2742800]
    baseline_params = [1010299, 1010955, 1010955, 1010955, 1489023, 2059635, 2722791]

    # Convert to kilo/mega for readability
    gnn_flops_k = [f / 1e3 for f in gnn_flops]
    baseline_flops_k = [f / 1e3 for f in baseline_flops]
    gnn_params_k = [p / 1e3 for p in gnn_params]
    baseline_params_k = [p / 1e3 for p in baseline_params]

    x = np.arange(len(domains))
    width = 0.35

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: FLOPs comparison
    ax1.bar(
        x - width / 2,
        gnn_flops_k,
        width,
        color=helper.MAP_COLOR["gnn"]["main"],
        label=helper.MAP_LABEL["gnn"],
    )
    ax1.bar(
        x + width / 2,
        baseline_flops_k,
        width,
        color=helper.MAP_COLOR["baseline"]["main"],
        label=helper.MAP_LABEL["baseline"],
    )

    ax1.set_xlabel("Domain")
    ax1.set_ylabel("FLOPs (thousands)")
    ax1.set_title("FLOPs per Forward Pass")
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains)
    ax1.legend()
    ax1.set_yscale("log")  # Log scale since difference is large

    # Plot 2: Parameters comparison
    ax2.bar(
        x - width / 2,
        gnn_params_k,
        width,
        color=helper.MAP_COLOR["gnn"]["main"],
        label=helper.MAP_LABEL["gnn"],
    )
    ax2.bar(
        x + width / 2,
        baseline_params_k,
        width,
        color=helper.MAP_COLOR["baseline"]["main"],
        label=helper.MAP_LABEL["baseline"],
    )

    ax2.set_xlabel("Domain")
    ax2.set_ylabel("Parameters (thousands)")
    ax2.set_title("Network Parameters")
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains)
    ax2.legend()
    ax2.set_yscale("log")  # Log scale since difference is large

    plt.tight_layout()
    helper.save_plot("network_stats.png")


def plot_ratio():
    """Plot the ratio of Baseline/GNN to show efficiency gain"""
    domains = ["slider", "red", "pink", "blue", "sr", "srp", "srpb"]

    gnn_flops = [125840, 125936, 125936, 125936, 161840, 197840, 233840]
    gnn_params = [12754, 13410, 13410, 13410, 13410, 13410, 13410]

    baseline_flops = [1021568, 1021664, 1021664, 1021664, 1502768, 2076512, 2742800]
    baseline_params = [1010299, 1010955, 1010955, 1010955, 1489023, 2059635, 2722791]

    flops_ratio = [b / g for b, g in zip(baseline_flops, gnn_flops)]
    params_ratio = [b / g for b, g in zip(baseline_params, gnn_params)]

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=helper.FIG_SIZE_FLAT)

    ax.bar(
        x - width / 2,
        flops_ratio,
        width,
        color=helper.MAP_COLOR["gnn"]["main"],
        label="FLOPs ratio (MLP/GNN)",
    )
    ax.bar(
        x + width / 2,
        params_ratio,
        width,
        color=helper.MAP_COLOR["tree"]["main"],
        label="Params ratio (MLP/GNN)",
    )

    ax.set_xlabel("Domain")
    ax.set_ylabel("Ratio (MLP / GNN)")
    ax.set_title("How many times smaller than MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()

    # Add horizontal line at 1 for reference
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    helper.save_plot("network_efficiency_ratio.png")
