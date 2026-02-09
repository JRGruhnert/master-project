import matplotlib.pyplot as plt
import src.plotting.helper as helper
from src.plotting.run import RunData, RunDataCollection


def plot(collection: RunDataCollection):
    """Plot training progress over time"""
    for run in collection.runs:
        batch_stats = run.stats["batch_stats"]

        epoch_indices = list(range(len(batch_stats)))
        episode_rewards = [batch["mean_episode_reward"] for batch in batch_stats]
        success_rates = [batch["success_rate"] for batch in batch_stats]
        max_success_rates = [batch["max_success_rate"] for batch in batch_stats]
        episode_lengths = [batch["mean_episode_length"] for batch in batch_stats]

        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=helper.FIG_SIZE)

        # Episode rewards
        ax1.plot(epoch_indices, episode_rewards, "b-", alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Reward")
        ax1.set_title("Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2.plot(epoch_indices, success_rates, "g-", alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Success Rate")
        ax2.set_title("Success Rates")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Episode lengths
        ax3.plot(epoch_indices, episode_lengths, "r-", alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean Length")
        ax3.set_title("Episode Lengths")
        ax3.grid(True, alpha=0.3)

        # Max success rates
        ax4.plot(epoch_indices, max_success_rates, "y-", alpha=0.7)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Success Rate")
        ax4.set_title("Max Success Rates")
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        helper.save_plot(f"{run.name}.png", subdir=f"{run.metadata['nt']}")
