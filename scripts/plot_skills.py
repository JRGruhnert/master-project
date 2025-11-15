import numpy as np
import matplotlib.pyplot as plt
import json


# Set a global style for all plots
print(plt.style.available)
plt.style.use("seaborn-v0_8")


def plot_skill_results(result_dict: dict[str, float]):
    """Plot skill evaluation results as stacked bar chart"""

    # Prepare data
    skills = list(result_dict.keys())
    success_rates = list(result_dict.values())
    failure_rates = [1.0 - rate for rate in success_rates]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar positions
    x = np.arange(len(skills))
    width = 0.6

    # Create stacked bars
    bars1 = ax.bar(x, success_rates, width, label="Success", color="#2E86AB")  # Blue
    bars2 = ax.bar(
        x, failure_rates, width, bottom=success_rates, label="Failure", color="#A23B72"
    )  # Red

    # Customize
    ax.set_ylabel("Rate", fontsize=12)
    ax.set_xlabel("Skills", fontsize=12)
    ax.set_title("Skill Evaluation Results", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(skills, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add percentage labels on bars
    for i, (success, failure) in enumerate(zip(success_rates, failure_rates)):
        if success > 0.05:  # Only show if bar is visible
            ax.text(
                i,
                success / 2,
                f"{success:.1%}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )
        if failure > 0.05:
            ax.text(
                i,
                success + failure / 2,
                f"{failure:.1%}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_skill_results_from_file(file: str):
    """Load results from JSON file and plot"""
    path = "results/skills/eval/" + file
    # ✅ Load JSON file
    with open(path, "r") as f:
        data = json.load(f)

    plot_skill_results(data)


def entry_point():
    # Usage:
    plot_skill_results_from_file("results.json")
