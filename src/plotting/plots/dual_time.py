import matplotlib.pyplot as plt
from src.plotting.helper import *
from src.plotting.run import RunData, RunDataCollection
import numpy as np


def smooth_data(data, window_size=5):
    """Smooth data using a moving average."""
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


def plot(collection: RunDataCollection):
    """Plot training progress over time"""
    for run1 in collection.runs:
        if run1.metadata["nt"] == "gnn":
            continue  # Skip GNN runs for this plot
        run2 = collection.get(
            nt="gnn",
            mode=run1.metadata["mode"],
            origin=run1.metadata["origin"],
            dest=run1.metadata["dest"],
            pe=run1.metadata["pe"],
            pr=run1.metadata["pr"],
        )

        batch_stats1 = run1.stats["batch_stats"]
        epoch_indices1 = list(range(len(batch_stats1)))
        episode_rewards1 = smooth_data(
            [batch["mean_episode_reward"] for batch in batch_stats1]
        )
        success_rates1 = smooth_data([batch["success_rate"] for batch in batch_stats1])
        max_success_rates1 = smooth_data(
            [batch["max_success_rate"] for batch in batch_stats1]
        )
        episode_lengths1 = smooth_data(
            [batch["mean_episode_length"] for batch in batch_stats1]
        )

        batch_stats2 = run2.stats["batch_stats"]
        epoch_indices2 = list(range(len(batch_stats2)))
        episode_rewards2 = smooth_data(
            [batch["mean_episode_reward"] for batch in batch_stats2]
        )
        success_rates2 = smooth_data([batch["success_rate"] for batch in batch_stats2])
        max_success_rates2 = smooth_data(
            [batch["max_success_rate"] for batch in batch_stats2]
        )
        episode_lengths2 = smooth_data(
            [batch["mean_episode_length"] for batch in batch_stats2]
        )

        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIG_SIZE)
        set_y_ticks(ax2)
        set_y_ticks(ax4)

        # Episode rewards
        ax1.plot(
            epoch_indices1,
            episode_rewards1,
            color=MAP_COLOR[NT_MLP]["main"],
            alpha=0.7,
            label=run1.name,
        )
        if run2:
            ax1.plot(
                epoch_indices2,
                episode_rewards2,
                color=MAP_COLOR[NT_GNN]["main"],
                alpha=0.7,
                label=run2.name,
            )
        ax1.set_xlabel(LABEL_EPOCH)
        ax1.set_ylabel(LABEL_REWARD)
        ax1.set_title("Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2.plot(
            epoch_indices1,
            success_rates1,
            color=MAP_COLOR[NT_MLP]["main"],
            alpha=0.7,
            label=run1.name,
        )
        if run2:
            ax2.plot(
                epoch_indices2,
                success_rates2,
                color=MAP_COLOR[NT_GNN]["main"],
                alpha=0.7,
                label=run2.name,
            )
        ax2.set_xlabel(LABEL_EPOCH)
        ax2.set_ylabel(LABEL_SR)
        ax2.set_title("Success Rates")
        ax2.set_ylim(0, 1)

        # Episode lengths
        ax3.plot(
            epoch_indices1,
            episode_lengths1,
            color=MAP_COLOR[NT_MLP]["main"],
            alpha=0.7,
            label=run1.name,
        )
        if run2:
            ax3.plot(
                epoch_indices2,
                episode_lengths2,
                color=MAP_COLOR[NT_GNN]["main"],
                alpha=0.7,
                label=run2.name,
            )
        ax3.set_xlabel(LABEL_EPOCH)
        ax3.set_ylabel(LABEL_LENGTH)
        ax3.set_title("Episode Lengths")
        ax3.grid(True, alpha=0.3)

        # Max success rates
        ax4.plot(
            epoch_indices1,
            max_success_rates1,
            color=MAP_COLOR[NT_MLP]["main"],
            alpha=0.7,
            label=run1.name,
        )
        if run2:
            ax4.plot(
                epoch_indices2,
                max_success_rates2,
                color=MAP_COLOR[NT_GNN]["main"],
                alpha=0.7,
                label=run2.name,
            )
        ax4.set_xlabel(LABEL_EPOCH)
        ax4.set_ylabel(LABEL_SR)
        ax4.set_title("Max Success Rates")
        ax4.set_ylim(0, 1)

        plt.title(f"Comparison of Training Progress")
        save_plot(
            f"{run1.name}{run2.name if run2 else ''}_comparison.png",
            subdir=f"comparison",
        )
