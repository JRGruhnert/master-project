import os
import glob
import re
from typing import Any
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set a global style for all plots
plt.style.use("seaborn-v0_8")


class RolloutAnalyzer:
    def __init__(self, path: str):
        """
        Initialize analyzer with path to directory containing .pt files

        Args:
            data_path: Path to directory containing stats_epoch_*.pt files
        """
        self.data_path = path + "/logs/"
        self.save_path = path
        self.summary_stats = self.compute_summary_stats()

    def load_all_batches(self) -> dict[int, dict]:
        """Load all rollout buffer files and return combined data"""
        # Updated pattern for new file format
        pattern = os.path.join(self.data_path, "stats_epoch_*.pt")
        files = glob.glob(pattern)

        if not files:
            print(f"No rollout buffer files found in {self.data_path}")
            print(f"Looking for pattern: stats_epoch_*.pt")
            # List all files in directory for debugging
            if os.path.exists(self.data_path):
                all_files = os.listdir(self.data_path)
                print(f"Available files: {all_files}")
            return {}

        print(f"Found {len(files)} rollout files")

        ready_data = {}
        for file_path in sorted(files):
            try:
                # Extract epoch number from filename
                filename = os.path.basename(file_path)
                epoch = int(filename.split("_")[-1].split(".")[0])

                # Load the .pt file
                data = torch.load(file_path, map_location="cpu")

                # Convert tensors to numpy for easier processing
                batch_data = {
                    "actions": data["actions"].numpy(),
                    "logprobs": data["logprobs"].numpy(),
                    "values": data["values"].numpy(),
                    "rewards": data["rewards"].numpy(),
                    "terminals": data["terminals"].numpy(),
                }

                ready_data[epoch] = batch_data
                print(f"Loaded epoch {epoch}: {len(batch_data['rewards'])} timesteps")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return ready_data

    def compute_batch_stats(self, batch_data: dict) -> dict:
        """Compute statistics for a single batch"""
        rewards = batch_data["rewards"]
        terminals = batch_data["terminals"]
        values = batch_data["values"]
        actions = batch_data["actions"]

        # Calculate episode statistics
        episode_rewards = []
        episode_lengths = []
        episode_successes = []

        current_episode_reward = 0
        current_episode_length = 0

        for i, (reward, terminal) in enumerate(zip(rewards, terminals)):
            current_episode_reward += reward
            current_episode_length += 1

            if terminal:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                # TODO: Remove hardcoded 50 as success threshold
                episode_successes.append(1 if reward >= 50.0 else 0)
                current_episode_reward = 0
                current_episode_length = 0

        return {
            "total_timesteps": len(rewards),
            "total_episodes": len(episode_rewards),
            "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0.0,
            "min_episode_reward": np.min(episode_rewards) if episode_rewards else 0.0,
            "max_episode_reward": np.max(episode_rewards) if episode_rewards else 0.0,
            "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0.0,
            "success_rate": np.mean(episode_successes) if episode_successes else 0.0,
            "mean_value": np.mean(values) if len(values) > 0 else 0.0,
            "mean_reward_per_step": np.mean(rewards) if len(rewards) > 0 else 0.0,
            "action_distribution": (
                np.bincount(actions.argmax(axis=1))
                if len(actions.shape) > 1
                else np.bincount(actions.astype(int))
            ),
        }

    def compute_summary_stats(self) -> dict:
        """Compute summary statistics across all batches"""
        batch_data = self.load_all_batches()

        # Check if we have any data
        if not batch_data:
            raise ValueError(
                "No batch data available for computing summary statistics."
            )
        # Compute stats for each batch
        batch_summaries = {}
        all_episode_rewards = []
        all_values = []
        all_success_rates = []
        all_episode_lengths = []
        total_timesteps = 0
        total_episodes = 0

        for epoch in sorted(batch_data.keys()):
            batch_stats = self.compute_batch_stats(batch_data[epoch])
            batch_summaries[epoch] = batch_stats

            # Collect data for overall stats
            rewards = batch_data[epoch]["rewards"]
            terminals = batch_data[epoch]["terminals"]
            values = batch_data[epoch]["values"]

            # Extract episode rewards for this batch
            current_episode_reward = 0
            for reward, terminal in zip(rewards, terminals):
                current_episode_reward += reward
                if terminal:
                    all_episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0

            all_values.extend(values)
            all_success_rates.append(batch_stats["success_rate"])
            all_episode_lengths.append(batch_stats["mean_episode_length"])
            total_timesteps += batch_stats["total_timesteps"]
            total_episodes += batch_stats["total_episodes"]

        # Overall statistics
        overall_stats = {
            "total_timesteps": total_timesteps,
            "total_batches": len(batch_data),
            "total_episodes": total_episodes,
            "mean_episode_reward": (
                np.mean(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "std_episode_reward": (
                np.std(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "min_episode_reward": (
                np.min(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "max_episode_reward": (
                np.max(all_episode_rewards) if all_episode_rewards else 0.0
            ),
            "mean_value": np.mean(all_values) if all_values else 0.0,
            "std_value": np.std(all_values) if all_values else 0.0,
            "mean_sr": (np.mean(all_success_rates) if all_success_rates else 0.0),
            "mean_episode_length": (
                np.mean(all_episode_lengths) if all_episode_lengths else 0.0
            ),
            "max_sr": (np.amax(all_success_rates) if all_success_rates else 0.0),
            "sr_until_max": (
                all_success_rates[: int(np.argmax(all_success_rates)) + 1]
                if all_success_rates
                else []
            ),
            "sr_until_90": (
                all_success_rates[
                    : int(np.argmax(np.array(all_success_rates) >= 0.9)) + 1
                ]
                if any(np.array(all_success_rates) >= 0.9)
                else []
            ),
            "sr_until_95": (
                all_success_rates[
                    : int(np.argmax(np.array(all_success_rates) >= 0.95)) + 1
                ]
                if any(np.array(all_success_rates) >= 0.95)
                else []
            ),
        }

        self.summary_stats = {
            "batch_summaries": batch_summaries,
            "overall": overall_stats,
        }

        return self.summary_stats

    def print_analysis(self):
        """Print comprehensive analysis of the rollout data"""
        print("\n" + "=" * 50)
        print("ROLLOUT ANALYSIS SUMMARY")
        print("=" * 50)

        overall = self.summary_stats["overall"]
        print(f"Total Timesteps: {overall['total_timesteps']:,}")
        print(f"Total Batches: {overall['total_batches']}")
        print(f"Total Episodes: {overall['total_episodes']}")
        print(
            f"Mean Episode Reward: {overall['mean_episode_reward']:.2f} ± {overall['std_episode_reward']:.2f}"
        )
        print(
            f"Episode Reward Range: [{overall['min_episode_reward']:.2f}, {overall['max_episode_reward']:.2f}]"
        )
        print(f"Mean Episode Length: {overall['mean_episode_length']:.1f} steps")
        print(f"Success Rate: {overall['mean_sr']:.1%}")
        print(
            f"Mean Value Estimate: {overall['mean_value']:.3f} ± {overall['std_value']:.3f}"
        )

        print("\n" + "-" * 30)
        print("PER-BATCH BREAKDOWN")
        print("-" * 30)

        batch_summaries = self.summary_stats["batch_summaries"]
        for epoch in sorted(batch_summaries.keys()):
            stats = batch_summaries[epoch]
            print(
                f"Epoch {epoch:3d}: "
                f"Episodes={stats['total_episodes']:2d}, "
                f"Reward={stats['mean_episode_reward']:6.1f}, "
                f"Success={stats['success_rate']:4.1%}, "
                f"Length={stats['mean_episode_length']:4.1f}"
            )

    def plot_training_curves(self):
        """Plot training progress over time"""
        batch_summaries = self.summary_stats["batch_summaries"]
        if not batch_summaries:
            print("No data to plot")
            return

        epochs = sorted(batch_summaries.keys())
        episode_rewards = [batch_summaries[e]["mean_episode_reward"] for e in epochs]
        success_rates = [batch_summaries[e]["success_rate"] for e in epochs]
        episode_lengths = [batch_summaries[e]["mean_episode_length"] for e in epochs]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        ax1.plot(epochs, episode_rewards, "b-", alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Episode Reward")
        ax1.set_title("Training Progress: Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2.plot(epochs, success_rates, "g-", alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Success Rate")
        ax2.set_title("Training Progress: Success Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Episode lengths
        ax3.plot(epochs, episode_lengths, "r-", alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean Episode Length")
        ax3.set_title("Training Progress: Episode Length")
        ax3.grid(True, alpha=0.3)

        # Combined plot
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, episode_rewards, "b-", label="Episode Reward")
        line2 = ax4_twin.plot(epochs, success_rates, "g-", label="Success Rate")

        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Episode Reward", color="b")
        ax4_twin.set_ylabel("Success Rate", color="g")
        ax4.set_title("Combined Training Progress")

        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc="center right")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(self.save_path, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def find_matching_rollout(
    target: dict, rollout_list: list[dict], match_keys: list[str] = ["tag", "pe", "pr"]
) -> dict | None:
    """Find rollout with matching key values"""
    for rollout in rollout_list:
        if all(rollout.get(key) == target.get(key) for key in match_keys):
            return rollout
    return None


def plot_agents_together_sr_vs_epoch(
    data: dict[str, dict[str, list[dict]]],
    save_path: str,
):
    """Compare all networks for same experiment parameters"""

    networks = ["gnn", "baseline", "search_tree"]
    colors = {"gnn": "blue", "baseline": "orange", "search_tree": "green"}

    # Use GNN as reference
    for ident, gnn_rollouts in data.get("gnn", {}).items():
        for gnn_rollout in gnn_rollouts:
            plt.figure(figsize=(12, 6))

            tag = gnn_rollout.get("tag")
            pe = gnn_rollout.get("pe")
            pr = gnn_rollout.get("pr")

            # ✅ Find and plot matching rollout from each network
            for network in networks:
                network_rollouts = data.get(network, {}).get(ident, [])

                matching_rollout = next(
                    (
                        r
                        for r in network_rollouts
                        if r.get("tag") == tag
                        and r.get("pe") == pe
                        and r.get("pr") == pr
                    ),
                    None,
                )

                if matching_rollout:
                    sr = matching_rollout.get("sr_until_max", [])
                    if sr:
                        plt.plot(
                            range(len(sr)),
                            sr,
                            label=network.upper(),
                            color=colors[network],
                            linewidth=2,
                            alpha=0.7,
                        )

            # Add threshold lines
            plt.axhline(
                y=0.9, color="orange", linestyle="--", alpha=0.5, label="90% threshold"
            )
            plt.axhline(
                y=0.95, color="red", linestyle="--", alpha=0.5, label="95% threshold"
            )

            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel("Success Rate", fontsize=12)
            plt.title(
                f"Network Comparison: {tag} (pe={pe}, pr={pr})",
                fontsize=14,
                fontweight="bold",
            )
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=11)
            plt.ylim(0, 1)
            plt.tight_layout()

            plot_path = os.path.join(save_path, f"compare_all_{tag}_pe{pe}_pr{pr}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()


"""
# agent vs epoch together
def plot_agents_together_sr_vs_epoch(
    data: dict[str, dict[str, list[dict[str, list[list[float] | float]]]]],
    save_path: str,
):
    # nt -> (t,r,e) -> tag,pe,pr,origin,dest,analzer values
    # nt -> gnn, baseline, search-tree

    for ident, gnn_rollouts in data["gnn"].items():
        baseline_rollouts = data.get("baseline", {}).get(ident, [])
        for gnn_rollout in gnn_rollouts:
            baseline_rollout = find_matching_rollout(gnn_rollout, baseline_rollouts)
            if baseline_rollout is None:
                continue

            plt.figure(figsize=(8, 5))
            plt.scatter(
                rollout[""],
                rollout[""],
                label="gnn",
            )

            plt.scatter(
                matching_rollout[""],
                rollout[""],
                label="baseline",
            )

            if label != "p":
                plt.scatter(
                    gnn_rollouts["p"],
                    y,
                    label=nt,
                )
        plt.xlabel("Epoch")
        plt.ylabel("%")
        plt.title(f"Success Rate vs {name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        plot_path = os.path.join(save_path, f"{nt}_{nt_data}_sr_vs_{name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

"""

"""
def plot_sr_vs_epoch_r(
    data: dict[str, dict[str, dict[str, dict[str, list[float] | float]]]],
    save_path: str,
):
    # data: gnn4 baseline1
    # ndata: t r e
    # idata: pe pr tag origin dest sr_until_max sr_until_90 sr_until_95 max_sr mean_sr
    # Define all origin-destination pairs

    pairs = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    for nt, n_data in data.items():
        for idx, (origin, dest) in enumerate(pairs):
            ident = "t"
            tag = f"{ident}{origin}{dest}"
            i_data = n_data.get(ident, {})
            i_data = [
                v
                for v, k in i_data.items()
                if isinstance(v, dict)
                and v.get("tag") == tag
                and v.get("pe") == 0.0
                and v.get("pr") == 0.0
            ]
            i_data.sort(key=lambda x: x.get("batch", 0))
            # Convert to dict of lists
            i_data = {key: [d.get(key) for d in i_data] for key in i_data[0]}

            sr_until_max = i_data.get("sr_until_max", [])
            start = sum(
                len(n_data[f"{ident}{o}{d}"].get("sr_until_max", []))
                for o, d in pairs[:idx]
            )
            end = start + len(sr_until_max)
            plt.figure(figsize=(8, 5))
            plt.scatter(
                range(start, end),
                sr_until_max,
                label=f"{tag}",
            )

            # Add horizontal threshold lines
        for y, color, label in [
            (0.9, "red", "90% Threshold"),
            (0.95, "orange", "95% Threshold"),
        ]:
            plt.axhline(y=y, color=color, linestyle="--", label=label)

        plt.xlabel("Epoch")
        plt.ylabel("SR %")
        plt.title(f"{nt} Retraining Success Rate by Transition")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(save_path, f"re_{nt}_retraining.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

"""


def plot_sr_vs_p(
    data: dict[str, dict[str, dict[str, list[float] | float]]],
    save_path: str,
    nt: str,
    tag: str,
):
    # Plot max success rate vs all p
    plt.figure(figsize=(8, 5))
    for name, rows in data[tag].items():
        print(rows.keys())
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=f"{name}_{label}",
                )
    plt.xlabel("p")
    plt.ylabel("Success Rate")
    plt.title(f"{nt} Success Rate vs p")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{nt}_{tag}_sr_vs_p.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def plot_sr_vs_p_comparison(
    data: dict[str, dict[str, list[dict]]],
    save_path: str,
):
    """Compare max_sr vs parameter across all networks for each tag"""

    networks = ["gnn", "baseline", "search_tree"]
    colors = {"gnn": "blue", "baseline": "orange", "search_tree": "green"}
    markers = {"gnn": "o", "baseline": "s", "search_tree": "^"}

    # Collect all unique tags across all networks
    all_tags = set()
    for network_data in data.values():
        for rollouts in network_data.values():
            for rollout in rollouts:
                all_tags.add(rollout.get("tag"))

    # Plot each tag with all networks
    for tag in sorted(all_tags):
        plt.figure(figsize=(12, 7))

        plotted_any = False
        param_name = None

        for network in networks:
            if network not in data:
                continue

            # Collect rollouts for this tag across all identifiers
            tag_rollouts = []
            for ident_rollouts in data[network].values():
                tag_rollouts.extend([r for r in ident_rollouts if r.get("tag") == tag])

            if not tag_rollouts:
                continue

            # Determine varying parameter
            pe_values = [r.get("pe", 0) for r in tag_rollouts]
            pr_values = [r.get("pr", 0) for r in tag_rollouts]

            pe_varies = len(set(pe_values)) > 1
            pr_varies = len(set(pr_values)) > 1

            if pe_varies:
                param_name = "pe"
                x_values = pe_values
            elif pr_varies:
                param_name = "pr"
                x_values = pr_values
            else:
                continue

            # Extract max_sr
            max_sr_values = [r.get("max_sr", 0) for r in tag_rollouts]

            # Sort
            sorted_pairs = sorted(zip(x_values, max_sr_values))
            x_sorted = [x for x, _ in sorted_pairs]
            y_sorted = [y for _, y in sorted_pairs]

            # ✅ Use plot instead of scatter
            plt.plot(
                x_sorted,
                y_sorted,
                marker=markers[network],
                markersize=10,
                linewidth=2.5,
                alpha=0.7,
                color=colors[network],
                markerfacecolor=colors[network],
                markeredgecolor="black",
                markeredgewidth=1.5,
                label=network.upper(),
            )

            plotted_any = True

        if plotted_any and param_name:
            # Add threshold lines
            plt.axhline(
                y=0.9,
                color="orange",
                linestyle="--",
                alpha=0.5,
                linewidth=2,
                label="90% Threshold",
            )
            plt.axhline(
                y=0.95,
                color="red",
                linestyle="--",
                alpha=0.5,
                linewidth=2,
                label="95% Threshold",
            )

            plt.xlabel(
                f"{param_name.upper()} (Parameter Value)",
                fontsize=13,
                fontweight="bold",
            )
            plt.ylabel("Maximum Success Rate", fontsize=13, fontweight="bold")
            plt.title(
                f"Network Comparison: {tag} - Max SR vs {param_name.upper()}",
                fontsize=15,
                fontweight="bold",
            )
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1.05)
            plt.legend(fontsize=11, loc="best")
            plt.tight_layout()

            # Save plot
            plot_path = os.path.join(
                save_path, tag, f"comparison_{tag}_maxsr_vs_{param_name}.png"
            )
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✅ Saved comparison: {plot_path}")


def plot_stats_direct(
    data: dict[str, dict[str, dict[str, list[float]]]],
    tag: str,
    save_path: str,
):
    plt.figure(figsize=(8, 5))
    for name, rows in data[tag].items():
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=label,
                )
    # Plot max success rate vs all p
    plt.figure(figsize=(8, 5))
    for name, rows in data[tag].items():
        x = rows["p"]
        for label, y in rows.items():
            if label != "p":
                plt.scatter(
                    x,
                    y,
                    label=label,
                )
    plt.xlabel("p")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs p")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{tag}_sr_vs_p.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def plot_retrain(
    x1: list[float],
    x2: list[float],
    y1: list[float],
    y2: list[float],
    name: str,
    save_path: str,
):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        x1,
        y1,
        label="P1",
    )
    plt.scatter(
        x2,
        y2,
        label="R1",
    )
    plt.xlabel("p (%)")
    plt.ylabel("Max Success Rate")
    plt.title("Max Success Rate vs p")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(save_path, f"{name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")


def entry_point():
    networks = ["gnn", "baseline", "search_tree"]
    t_tags = ["t1", "t2", "t3"]
    # r_tags = ["r12", "r21", "r13", "r23", "r31", "r32"]
    r_tags = []
    e_tags = ["e11", "e12", "e13", "e21", "e22", "e23", "e31", "e32", "e33"]
    tags = t_tags + r_tags + e_tags
    result_path = f"results/plots"
    # Create directories if they don't exist
    for tag in tags:
        os.makedirs(result_path + f"/{tag}", exist_ok=True)

    all_data: dict[str, Any] = {}
    for nt in networks:
        read_path = f"results/{nt}/"
        all_results = glob.glob(f"{read_path}/*", recursive=True)

        file_pattern = re.compile(
            rf"(?P<tag>{'|'.join(tags)})_pe_(?P<pe>[0-9.]+)_pr_(?P<pr>[0-9.]+)"
        )
        tag_pattern = re.compile(
            rf"(?P<ident>{'|'.join(['t', 'r', 'e'])})?(?P<origin>\d)(?P<dest>\d)?"
        )

        data = {"t": [], "r": [], "e": []}
        for path in all_results:
            file_match = file_pattern.search(path)
            if file_match:
                analyzer = RolloutAnalyzer(path)
                analyzer.print_analysis()
                analyzer.plot_training_curves()
                tag_match = tag_pattern.search(file_match.group("tag"))
                if tag_match:
                    data[tag_match.group("ident")].append(
                        {
                            **analyzer.summary_stats["overall"],
                            "pe": float(file_match.group("pe")),
                            "pr": float(file_match.group("pr")),
                            "origin": tag_match.group("origin"),
                            "dest": (
                                tag_match.group("dest")
                                if tag_match.group("dest")
                                else tag_match.group("origin")
                            ),
                            "tag": file_match.group("tag"),  # for searching
                        }
                    )
        all_data[nt] = data
    plot_agents_together_sr_vs_epoch(all_data, result_path)  # type: ignore
    plot_sr_vs_p_comparison(all_data, result_path)
    # plot_sr_vs_p(all_data, result_path)
    # plot_sr_vs_epoch_r(all_data, result_path)
    # plot_sr_vs_epoch_r(data, result_path, network)
    # plot_sr_vs_epoch_r(data, result_path, network)
    # plot_sr_vs_category(all_data, result_path)
