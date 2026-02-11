import matplotlib.pyplot as plt
from src.plotting.helper import *
from src.plotting.run import RunData, RunDataCollection
import numpy as np


def plot(collection: RunDataCollection):
    """Plot training progress over time"""
    run_s_g = collection.get(
        nt="gnn", mode="s", origin="srpb", dest="sr", pe=0.0, pr=0.0
    )
    run_s_b = collection.get(
        nt="baseline", mode="s", origin="srpb", dest="sr", pe=0.0, pr=0.0
    )
    run_t_g = collection.get(
        nt="gnn", mode="t", origin="srpb", dest="sr", pe=0.0, pr=0.0
    )
    run_t_b = collection.get(
        nt="baseline", mode="t", origin="srpb", dest="sr", pe=0.0, pr=0.0
    )

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIG_SIZE)
    set_y_ticks(ax2)
    set_y_ticks(ax4)

    for run in [
        {"run": run_s_g, "name": "GNN Sparse", "color": MAP_COLOR[NT_GNN]["main"]},
        {"run": run_s_b, "name": "MLP Sparse", "color": MAP_COLOR[NT_MLP]["main"]},
        {"run": run_t_g, "name": "GNN Dense", "color": MAP_COLOR[NT_GNN]["main"]},
        {"run": run_t_b, "name": "MLP Dense", "color": MAP_COLOR[NT_MLP]["main"]},
    ]:
        batch_stats = run["run"].stats["batch_stats"]
        epoch_indices = list(range(len(batch_stats)))
        episode_rewards = smooth_data(
            [batch["mean_episode_reward"] for batch in batch_stats]
        )
        success_rates = smooth_data([batch["success_rate"] for batch in batch_stats])
        max_success_rates = smooth_data(
            [batch["max_success_rate"] for batch in batch_stats]
        )
        episode_lengths = smooth_data(
            [batch["mean_episode_length"] for batch in batch_stats]
        )

        # Episode rewards
        ax1.plot(
            epoch_indices,
            episode_rewards,
            color=run["color"],
            alpha=0.7,
            label=run["name"],
        )
        ax1.set_xlabel(LABEL_EPOCH)
        ax1.set_ylabel(LABEL_REWARD)
        ax1.set_title("Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2.plot(
            epoch_indices,
            success_rates,
            color=run["color"],
            alpha=0.7,
            label=run["name"],
        )

        ax2.set_xlabel(LABEL_EPOCH)
        ax2.set_ylabel(LABEL_SR)
        ax2.set_title("Success Rates")
        ax2.set_ylim(0, 1)

        # Episode lengths
        ax3.plot(
            epoch_indices,
            episode_lengths,
            color=run["color"],
            alpha=0.7,
            label=run["name"],
        )

        ax3.set_xlabel(LABEL_EPOCH)
        ax3.set_ylabel(LABEL_LENGTH)
        ax3.set_title("Episode Lengths")
        ax3.grid(True, alpha=0.3)

        # Max success rates
        ax4.plot(
            epoch_indices,
            max_success_rates,
            color=run["color"],
            alpha=0.7,
            label=run["name"],
        )

        ax4.set_xlabel(LABEL_EPOCH)
        ax4.set_ylabel(LABEL_SR)
        ax4.set_title("Max Success Rates")
        ax4.set_ylim(0, 1)

    handles, labels = (
        ax1.get_legend_handles_labels()
    )  # Get the legend handles and labels from one of the axes
    plt.legend(handles, labels)  # Adjust location and layout
    plt.title(f"Comparison of Reward Modes on the SR SKill Set")
    save_plot(f"reward_mode_comparison.png")
