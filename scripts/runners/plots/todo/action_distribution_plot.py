import matplotlib.pyplot as plt
import numpy as np
from heca.misc.run_data import TrainingRunData, RunDataCollection
from heca.plots.helper.helper import *

dom = [
    ["blue", "slider"],
    ["slider", "blue"],
    ["blue", "pink"],
    ["pink", "blue"],
    ["blue", "red"],
    ["red", "blue"],
]


def plot(collection: RunDataCollection):
    for nt in [NT_GNN, NT_MLP]:
        for index, domain in enumerate(
            [
                ["blue", "slider"],
                ["slider", "red"],
                ["red", "pink"],
                ["pink", "blue"],
                ["red", "green"],
                ["pink", "yellow"],
            ]
        ):
            run_g_t = collection.get(
                nt=NT_GNN,
                mode=MODE_EVAL,
                origin=domain[0],
                dest=domain[0],
                pe=0.0,
                pr=0.0,
            )
            run_g_d = collection.get(
                nt=NT_GNN,
                mode=MODE_DOMAIN,
                origin=domain[1],
                dest=domain[1],
                pe=0.0,
                pr=0.0,
            )
            run_b_t = collection.get(
                nt=NT_MLP,
                mode=MODE_EVAL,
                origin=domain[0],
                dest=domain[0],
                pe=0.0,
                pr=0.0,
            )
            run_b_d = collection.get(
                nt=NT_MLP,
                mode=MODE_DOMAIN,
                origin=domain[1],
                dest=domain[1],
                pe=0.0,
                pr=0.0,
            )
            g_d_actions_list = run_g_d.stats["batch_stats"][-1]["action_distribution"]
            g_t_actions_list = run_g_t.stats["batch_stats"][-1]["action_distribution"]
            b_d_actions_list = run_b_d.stats["batch_stats"][-1]["action_distribution"]
            b_t_actions_list = run_b_t.stats["batch_stats"][-1]["action_distribution"]
            x = np.arange(len(g_d_actions_list))
            width = 0.2

            fig, ax = plt.subplots(figsize=FIG_SIZE_FLAT)

            g_data = np.array(g_d_actions_list) - np.array(g_t_actions_list)
            b_data = np.array(b_d_actions_list) - np.array(b_t_actions_list)
            ax.bar(
                x + width / 2,
                g_data,
                width,
                color=MAP_COLOR[NT_GNN]["main"],
                label=f"{MAP_LABEL[NT_GNN]}",
            )
            ax.bar(
                x - width / 2,
                b_data,
                width,
                color=MAP_COLOR[NT_MLP]["main"],
                label=f"{MAP_LABEL[NT_MLP]}",
            )

            # Calculate delta success rate and episode length
            g_d_sr = run_g_d.stats["batch_stats"][-1]["success_rate"]
            g_t_sr = run_g_t.stats["batch_stats"][-1]["success_rate"]
            b_d_sr = run_b_d.stats["batch_stats"][-1]["success_rate"]
            b_t_sr = run_b_t.stats["batch_stats"][-1]["success_rate"]

            g_d_len = run_g_d.stats["batch_stats"][-1]["mean_episode_length"]
            g_t_len = run_g_t.stats["batch_stats"][-1]["mean_episode_length"]
            b_d_len = run_b_d.stats["batch_stats"][-1]["mean_episode_length"]
            b_t_len = run_b_t.stats["batch_stats"][-1]["mean_episode_length"]

            g_d_success_len = run_g_d.stats["batch_stats"][-1][
                "mean_success_episode_length"
            ]
            g_t_success_len = run_g_t.stats["batch_stats"][-1][
                "mean_success_episode_length"
            ]
            b_d_success_len = run_b_d.stats["batch_stats"][-1][
                "mean_success_episode_length"
            ]
            b_t_success_len = run_b_t.stats["batch_stats"][-1][
                "mean_success_episode_length"
            ]

            g_d_failure_len = run_g_d.stats["batch_stats"][-1][
                "mean_failure_episode_length"
            ]
            g_t_failure_len = run_g_t.stats["batch_stats"][-1][
                "mean_failure_episode_length"
            ]
            b_d_failure_len = run_b_d.stats["batch_stats"][-1][
                "mean_failure_episode_length"
            ]
            b_t_failure_len = run_b_t.stats["batch_stats"][-1][
                "mean_failure_episode_length"
            ]

            def color_text(value, text):
                color = "green" if value >= 0 else "red"
                return f"$\\textcolor{{{color}}}{{{text}}}$"

            legend_text = (
                "Metric      GNN (Δ%)         MLP (Δ%)\n"
                "SR          {:>2.3f} ({:>+2.1f}%)   {:>2.3f} ({:>+2.1f}%)\n"
                "EpLen       {:>2.3f} ({:>+2.1f}%)   {:>2.3f} ({:>+2.1f}%)\n"
                "SuccessLen  {:>2.3f} ({:>+2.1f}%)   {:>2.3f} ({:>+2.1f}%)\n"
                "FailureLen  {:>2.3f} ({:>+2.1f}%)   {:>2.3f} ({:>+2.1f}%)"
            ).format(
                g_t_sr,
                100 * (g_d_sr - g_t_sr) / max(1e-6, g_t_sr),
                b_t_sr,
                100 * (b_d_sr - b_t_sr) / max(1e-6, b_t_sr),
                g_t_len,
                100 * (g_d_len - g_t_len) / max(1e-6, g_t_len),
                b_t_len,
                100 * (b_d_len - b_t_len) / max(1e-6, b_t_len),
                g_t_success_len,
                100 * (g_d_success_len - g_t_success_len) / max(1e-6, g_t_success_len),
                b_t_success_len,
                100 * (b_d_success_len - b_t_success_len) / max(1e-6, b_t_success_len),
                g_t_failure_len,
                100 * (g_d_failure_len - g_t_failure_len) / max(1e-6, g_t_failure_len),
                b_t_failure_len,
                100 * (b_d_failure_len - b_t_failure_len) / max(1e-6, b_t_failure_len),
            )

            # Add legend box to plot
            ax.text(
                0.45,
                0.85,
                legend_text,
                transform=ax.transAxes,
                fontsize=14,
                verticalalignment="top",
                horizontalalignment="left",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.7),
            )

            ax.axhline(0, linewidth=1)
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels([f"{i}" for i in range(len(g_data))])
            ax.set_ylabel("Delta Actions")
            ax.set_xlabel("Skill Index")
            ax.set_title(
                f"Delta Action Distribution: {dom[index][0]} vs {dom[index][1]} (trained on {dom[index][0]})"
            )
            ax.legend()
            save_plot(f"{dom[index][0]}_to_{dom[index][1]}.png")
