import numpy as np
import matplotlib.pyplot as plt
import json


# Set a global style for all plots
plt.style.use("seaborn-v0_8")


def plot_skill_results(result_dict: dict[str, float]):
    """Plot skill evaluation results as stacked bar chart"""

    # Prepare data
    skills = list(result_dict.keys())
    success_rates = list(result_dict.values())
    failure_rates = [1.0 - rate for rate in success_rates]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar positions with gap after CloseSlideBack
    x = np.arange(len(skills), dtype=float)
    gap_after_skill = "CloseSlideBack"

    # Find the index of CloseSlideBack and add gap after it
    if gap_after_skill in skills:
        gap_index = skills.index(gap_after_skill)
        # Add 0.8 extra spacing to all bars after CloseSlideBack to create a visible gap
        for i in range(gap_index + 1, len(x)):
            x[i] += 0.8

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
    legend = ax.legend(
        frameon=True, facecolor="white", edgecolor="gray", framealpha=0.9
    )
    ax.grid(axis="y", alpha=0.3)

    # Add float labels only on failure bars
    for i, failure in enumerate(failure_rates):
        if failure > 0.05:  # Only show if bar is visible enough
            ax.text(
                x[i],  # Use the modified x position instead of i
                success_rates[i] + failure / 2,
                f"{failure:.2f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    plt.tight_layout()
    plt.show()


def plot_skill_results_from_file(file: str):
    """Load results from JSON file and plot"""
    path = "results/skills/eval/plots/" + file
    # ✅ Load JSON file
    with open(path, "r") as f:
        data = json.load(f)

    plot_skill_results(data)


def entry_point():
    # Usage:
    plot_skill_results_from_file("results.json")
