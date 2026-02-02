import os
import glob
import re
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set a global style for all plots
plt.style.use("seaborn-v0_8")
COLORS = {"gnn": "blue", "baseline": "orange", "search_tree": "green"}
SAVE_PATH = "plots"

class RunData:
    def __init__(self, path: str, metadata: dict):
        """
        Initialize analyzer with path to directory containing .pt files

        Args:
            data_path: Path to directory containing stats_epoch_*.pt files
        """
        self.data_path = path + "/logs/"
        self.save_path = path
        self.metadata = metadata
        self.stats = self.compute_stats()
        self.print_analysis()
        self.plot_training_curves()

    def load_all_batches(self) -> list[dict]:
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
            return []  # Return empty list if no files found

        print(f"Found {len(files)} rollout files")

        run_data = []
        for file_path in sorted(files):
            try:
                # Extract epoch number from filename
                filename = os.path.basename(file_path)
                epoch = int(filename.split("_")[-1].split(".")[0])

                # Load the .pt file
                loaded_data = torch.load(file_path, map_location="cpu")

                # Convert tensors to numpy for easier processing
                epoch_data = {
                    "actions": loaded_data["actions"].numpy(),
                    "logprobs": loaded_data["logprobs"].numpy(),
                    "values": loaded_data["values"].numpy(),
                    "rewards": loaded_data["rewards"].numpy(),
                    "success": loaded_data["success"].numpy(),
                    "terminals": loaded_data["terminals"].numpy(),
                }

                run_data[epoch] = epoch_data
                print(f"Loaded epoch {epoch}: {len(epoch_data['rewards'])} timesteps")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return run_data

    def compute_batch_stats(self, batch_data: dict[str, np.ndarray]) -> dict:
        """Compute statistics for a single batch"""
        # values = batch_data["values"]
        actions = batch_data["actions"]
        rewards: list[float] = batch_data["rewards"].tolist()
        success: list[bool] = batch_data["success"].tolist()
        terminals: list[bool] = batch_data["terminals"].tolist()

        batch_size = len(rewards)

        return {
            "batch_size": batch_size,
            "total_episodes": max(1, terminals.count(True)),
            "mean_episode_length": batch_size / max(1, terminals.count(True)),
            "total_rewards": sum(rewards),
            "mean_episode_reward": sum(rewards) / max(1, terminals.count(True)),
            "success_rate": success.count(True) / max(1, terminals.count(True)),
            "successes": success.count(True),
            "action_distribution": (
                np.bincount(actions.argmax(axis=1))
                if len(actions.shape) > 1
                else np.bincount(actions.astype(int))
            ),
        }

    def compute_stats(self) -> dict:
        """Compute summary statistics across all batches"""
        all_batch_stats = self.load_all_batches()

        # Check if we have any data
        if not all_batch_stats:
            raise ValueError(
                "No batch data available for computing summary statistics."
            )
        # Compute stats for each batch
        run_episode_rewards = []
        all_success_rates = []
        all_episode_lengths = []
        total_timesteps = 0
        total_episodes = 0

        for value in all_batch_stats:
            batch_stats = self.compute_batch_stats(value)

            # Collect data for overall statss
            rewards = value["rewards"]
            terminals = value["terminals"]

            # Extract episode rewards for this batch
            current_episode_reward = 0
            for reward, terminal in zip(rewards, terminals):
                current_episode_reward += reward
                if terminal:
                    run_episode_rewards.append(current_episode_reward)
                    current_episode_reward = 0

            all_success_rates.append(batch_stats["success_rate"])
            all_episode_lengths.append(batch_stats["mean_episode_length"])
            total_timesteps += batch_stats["total_timesteps"]
            total_episodes += batch_stats["total_episodes"]

        # Overall statistics
        total_timesteps = sum(d["total_timesteps"] for d in all_batch_stats)
        total_episodes = sum(d["total_episodes"] for d in all_batch_stats)
        total_rewards = sum(d["total_rewards"] for d in all_batch_stats)
        total_successes = sum(d["successes"] for d in all_batch_stats)
        max_success_rate = max(d["success_rate"] for d in all_batch_stats)
        index_of_max = int(np.argmax([d["success_rate"] for d in all_batch_stats]))
        index_of_first_90 = (
            int(np.argmax(np.array(all_success_rates) >= 0.9))
            if any(np.array(all_success_rates) >= 0.9)
            else None
        )
        index_of_first_95 = (
            int(np.argmax(np.array(all_success_rates) >= 0.95))
            if any(np.array(all_success_rates) >= 0.95)
            else None
        )

        overall_stats = {
            "total_timesteps": total_timesteps,
            "total_batches": len(all_batch_stats),
            "total_episodes": total_episodes,
            "mean_episode_reward": total_rewards / max(1, total_episodes),
            "mean_sr": total_successes / max(1, total_episodes),
            "mean_episode_length": total_timesteps / max(1, total_episodes),
            "max_sr": max_success_rate,
            "sr_until_max": all_success_rates[: index_of_max + 1],
            "sr_until_90": (
                all_success_rates[: index_of_first_90 + 1]
                if index_of_first_90 is not None
                else []
            ),
            "sr_until_95": (
                all_success_rates[: index_of_first_95 + 1]
                if index_of_first_95 is not None
                else []
            ),
        }

        return {
            "batch_stats": all_batch_stats,
            "run_stats": overall_stats,
        }

    def print_analysis(self):
        """Print comprehensive analysis of the rollout data"""
        print("\n" + "=" * 50)
        print("ROLLOUT ANALYSIS SUMMARY")
        print("=" * 50)

        overall = self.stats["run_stats"]
        print(f"Total Timesteps: {overall['total_timesteps']:,}")
        print(f"Total Batches: {overall['total_batches']}")
        print(f"Total Episodes: {overall['total_episodes']}")
        print(f"Mean Episode Reward: {overall['mean_episode_reward']:.2f}")
        print(f"Mean Episode Length: {overall['mean_episode_length']:.1f} steps")
        print(f"Success Rate: {overall['mean_sr']:.1%}")
        print(f"Mean Value Estimate: {overall['mean_value']:.3f}")

        print("\n" + "-" * 30)
        print("PER-BATCH BREAKDOWN")
        print("-" * 30)

        for idx, batch in enumerate(self.stats["batch_stats"]):
            print(
                f"Epoch {idx:3d}: "
                f"Episodes={batch['total_episodes']:2d}, "
                f"Reward={batch['mean_episode_reward']:6.1f}, "
                f"Success={batch['success_rate']:4.1%}, "
                f"Length={batch['mean_episode_length']:4.1f}"
            )

    def plot_training_curves(self):
        """Plot training progress over time"""
        batch_stats = self.stats["batch_stats"]
        if not batch_stats:
            print("No data to plot")
            return

        epoch_indices = list(range(len(batch_stats)))
        episode_rewards = [batch["mean_episode_reward"] for batch in batch_stats]
        success_rates = [batch["success_rate"] for batch in batch_stats]
        episode_lengths = [batch["mean_episode_length"] for batch in batch_stats]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        ax1.plot(epoch_indices, episode_rewards, "b-", alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Episode Reward")
        ax1.set_title("Training Progress: Episode Rewards")
        ax1.grid(True, alpha=0.3)

        # Success rate
        ax2.plot(epoch_indices, success_rates, "g-", alpha=0.7)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Success Rate")
        ax2.set_title("Training Progress: Success Rate")
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Episode lengths
        ax3.plot(epoch_indices, episode_lengths, "r-", alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Mean Episode Length")
        ax3.set_title("Training Progress: Episode Length")
        ax3.grid(True, alpha=0.3)

        # Combined plot
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epoch_indices, episode_rewards, "b-", label="Episode Reward")
        line2 = ax4_twin.plot(epoch_indices, success_rates, "g-", label="Success Rate")

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

class RunDataCollection:
    def __init__(self):
        self.runs: list[RunData] = []

    def add(self, run: RunData):
        self.runs.append(run)

    def get(self, nt: str, mode: str, origin: str, dest: str, pe: float, pr: float) -> RunData | None:
        """Filter runs based on a predicate function"""
        for run in self.runs:
            if (
                run.metadata.get("nt") == nt
                and run.metadata.get("mode") == mode
                and run.metadata.get("origin") == origin
                and run.metadata.get("dest") == dest
                and run.metadata.get("pe") == pe
                and run.metadata.get("pr") == pr
            ):
                return run


def plot_threshold_lines():
    # Add threshold lines
    plt.axhline(
        y=0.9, color="orange", linestyle="--", alpha=0.5, label="90% threshold"
    )
    plt.axhline(
        y=0.95, color="red", linestyle="--", alpha=0.5, label="95% threshold"
    )

def save_plot(filename: str):
    """Save plot to specified path"""
    os.makedirs(SAVE_PATH, exist_ok=True)
    plot_path = os.path.join(SAVE_PATH, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Plot: {plot_path}")

def plot_p_comparison(
    data: dict[str, dict[str, list[float]]]
):
    """Compare max_sr vs parameter across all networks for each tag
    Args:
        data: dict[tag][network] = p_tag, network -> list of max_sr
    
    """
    

    x_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Plot each tag with all networks
    for p_tag in data.keys():
        plt.figure(figsize=(12, 7))
        for network in data[p_tag].keys():
            plt.plot(
                x_values,
                data[p_tag][network],
                markersize=10,
                linewidth=2.5,
                alpha=0.7,
                color=COLORS[network],
                markeredgewidth=1.5,
                label=network.upper(),
            )

        plot_threshold_lines()

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
        plt.tight_layout()

        save_plot(f"comparison_{p_tag}.png")






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
            plot_threshold_lines()

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

        overall_stats = {
            "total_timesteps": total_timesteps,
            "total_batches": len(all_batch_stats),
            "total_episodes": total_episodes,
            "mean_episode_reward": total_rewards / max(1, total_episodes),
            "mean_sr": total_successes / max(1, total_episodes),
            "mean_episode_length": total_timesteps / max(1, total_episodes),
            "max_sr": max_success_rate,
            "sr_until_max": all_success_rates[: index_of_max + 1],
            "sr_until_90": (
                all_success_rates[: index_of_first_90 + 1]
                if index_of_first_90 is not None
                else []
            ),
            "sr_until_95": (
                all_success_rates[: index_of_first_95 + 1]
                if index_of_first_95 is not None
                else []
            ),
        }

        return {
            "batch_stats": all_batch_stats,
            "run_stats": overall_stats,
        }

def make_plots(runs: RunDataCollection):
    """Generate plots for all runs in the collection"""
    p_data: dict[str, dict[str, list[float]]] = {}
    for p in ["pe", "pr"]:
        for nt in ["gnn", "baseline"]:
            for value in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                # Collect data for plotting
                run: RunData | None = None
                if p == "pe":
                    run =runs.get(nt=nt, mode="t", origin="b", dest="b", pe=value, pr=0.0)
                else:
                    run =runs.get(nt=nt, mode="t", origin="b", dest="b", pe=0.0, pr=value)
                if run is not None:
                    max_sr = run.stats["run_stats"]["max_sr"]
                    p_data[p][nt].append(max_sr)
                else:
                    print(f"Run not found for {nt} with {p}={value}")
    plot_p_comparison(p_data)


    domain_data: dict[str, dict[str, list[float]]] = {}
    for nt in ["gnn", "baseline"]:
        for domain in ["blue", "red", "pink", "slide"]:
            run_t = runs.get(nt=nt, mode="t", origin=domain, dest=domain, pe=0.0, pr=0.0)
            run_e = runs.get(nt=nt, mode="e", origin=domain, dest=domain, pe=0.0, pr=0.0)
            if run_t is not None and run_e is not None:
                max_sr_t = run_t.stats["run_stats"]["max_sr"]
                max_sr_e = run_e.stats["run_stats"]["max_sr"]
                print(f"Domain: {domain}, Network: {nt}, Training SR: {max_sr_t:.3f}, Eval SR: {max_sr_e:.3f}")
                
        max_sr = run.stats["run_stats"]["max_sr"]
        p_data[nt][domain].append(max_sr)


def entry_point():
    networks = ["gnn", "baseline", "tree"]
    training_tags = [
        "t_b_b",
        "t_sr_sr",
        "t_sp_sp",
        "t_sb_sb",
        "t_brpb_br",
        "t_brpb_brp",
        "t_brpb_brpb",
    ]
    retraining_tags = [
        "r_brpb_brp",
        "r_brpb_brpb",
    ]
    eval_tags = [
        "e_b_b",
        "e_sr_sr",
        "e_sp_sp",
        "e_sb_sb",
    ]
    tags = training_tags + retraining_tags + eval_tags
    collection = RunDataCollection()
    for nt in networks:
        read_path = f"results/{nt}/"
        all_results = glob.glob(f"{read_path}/*", recursive=True)

        file_pattern = re.compile(
            rf"(?P<tag>{'|'.join(tags)})_pe(?P<pe>[0-9.]+)_pr(?P<pr>[0-9.]+)"
        )
        tag_pattern = re.compile(
            rf"(?P<ident>{'|'.join(['t', 'r', 'e'])})?(?P<origin>\d)(?P<dest>\d)?"
        )

        for path in all_results:
            file_match = file_pattern.search(path)
            if file_match:
                tag_match = tag_pattern.search(file_match.group("tag"))
                if tag_match:
                    metadata = {
                        "nt": nt,
                        "mode": file_match.group("tag"),
                        "pe": float(file_match.group("pe")),
                        "pr": float(file_match.group("pr")),
                        "origin": tag_match.group("origin"),
                        "dest": tag_match.group("dest"),
                    }
                    # Collect data for further analysis
                    collection.add(RunData(path, metadata))

    make_plots(collection)
