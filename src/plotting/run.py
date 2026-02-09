import os
import glob
import torch
import numpy as np


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
        for file_path in sorted(
            files, key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0])
        ):
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

                run_data.append(epoch_data)
                # print(f"Loaded epoch {epoch}: {len(epoch_data['rewards'])} timesteps")

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        return run_data

    def compute_batch_stats(
        self,
        batch_data: dict[str, np.ndarray],
        prev_sr=0.0,
    ) -> dict:
        """Compute statistics for a single batch"""
        # values = batch_data["values"]
        actions = batch_data["actions"]
        rewards: list[float] = batch_data["rewards"].tolist()
        success: list[bool] = batch_data["success"].tolist()
        terminals: list[bool] = batch_data["terminals"].tolist()

        batch_size = len(rewards)
        # Handle action distribution with -1 as special category
        if len(actions.shape) > 1:
            action_indices = actions.argmax(axis=1)
        else:
            action_indices = actions.astype(int)

        # Count -1s separately, then bincount the rest
        negative_count = np.sum(action_indices == -1)
        positive_actions = action_indices[action_indices >= 0]

        if len(positive_actions) > 0:
            action_distribution = np.bincount(positive_actions).tolist()
        else:
            action_distribution = []

        # Append -1 count at the end
        action_distribution.append(int(negative_count))

        sr = success.count(True) / max(1, terminals.count(True))

        return {
            "batch_size": batch_size,
            "total_episodes": max(1, terminals.count(True)),
            "mean_episode_length": batch_size / max(1, terminals.count(True)),
            "total_rewards": sum(rewards),
            "mean_episode_reward": sum(rewards) / max(1, terminals.count(True)),
            "success_rate": sr,
            "max_success_rate": max(sr, prev_sr),
            "successes": success.count(True),
            "action_distribution": action_distribution,
        }

    def compute_stats(self) -> dict:
        """Compute summary statistics across all batches"""
        # Compute stats for each batch
        all_success_rates = []
        all_episode_lengths = []
        total_timesteps = 0
        total_episodes = 0
        total_rewards = 0.0
        total_successes = 0
        max_success_rate = 0.0

        computed_batch_stats = []
        all_batch_data = self.load_all_batches()
        # Check if we have any data
        if not all_batch_data:
            raise ValueError(
                "No batch data available for computing summary statistics."
            )
        prev_sr = 0.0
        for idx, batch_data in enumerate(all_batch_data):
            batch_stats = self.compute_batch_stats(batch_data, prev_sr=prev_sr)
            prev_sr = batch_stats["max_success_rate"]

            computed_batch_stats.append(batch_stats)
            all_success_rates.append(batch_stats["success_rate"])
            all_episode_lengths.append(batch_stats["mean_episode_length"])
            total_timesteps += batch_stats["batch_size"]
            total_episodes += batch_stats["total_episodes"]
            total_rewards += batch_stats["total_rewards"]
            total_successes += batch_stats["successes"]
            max_success_rate = max(max_success_rate, batch_stats["success_rate"])

        index_of_max = int(np.argmax(all_success_rates))
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
            "total_batches": len(all_batch_data),
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
            "batch_stats": computed_batch_stats,
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

    @property
    def name(
        self,
    ) -> str:
        """Generate a descriptive name for this run based on metadata"""
        return (
            f"{self.metadata.get('mode', 'err')}_"
            + f"{self.metadata.get('origin', 'err')}_"
            + f"{self.metadata.get('dest', 'err')}_"
            + f"pe{self.metadata.get('pe', 'err')}_"
            + f"pr{self.metadata.get('pr', 'err')}"
        )


class RunDataCollection:
    def __init__(self):
        self.runs: list[RunData] = []

    def add(self, run: RunData):
        self.runs.append(run)

    def get(
        self, nt: str, mode: str, origin: str, dest: str, pe: float, pr: float
    ) -> RunData:
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
        else:
            raise ValueError(
                f"No run found for network={nt}, mode={mode}, origin={origin}, dest={dest}, pe={pe}, pr={pr}"
            )
