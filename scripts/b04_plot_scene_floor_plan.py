import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import argparse

from heca.misc import logger
from heca.scenes.scene import Scene

from scripts.b03_plot_tapas_models import get_env_safely
from scripts.common.args import add_scene_argument
from scripts.common.scenes import iter_scene_configs

# Style by object class (get_object_boundaries() returns no colors).
COLORS = {
    "cube": "red",
    "lid": "yellow",
    "peg": "green",
    "faucet": "orange",
    "drawer": "blue",
    "window": "purple",
    "slider": "cyan",
    "box": "brown",
    "shelf": "goldenrod",
    "button": "gray",
}


def _color(s, name):
    base = "".join(ch for ch in name if not ch.isdigit())  # cube0 -> cube
    return COLORS.get(base, "0.4")


def plot_scene_boundaries(
    scene: Scene,
    ax=None,
    alpha=0.3,
    show_spawn_outline=True,
    label=True,
    label_kwargs=None,
    margin=0.12,
):
    env = get_env_safely(scene)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    geo = env.get_object_boundaries()

    wb = env._workspace_bounds
    ax.add_patch(
        Rectangle(
            wb[0],
            wb[1][0] - wb[0][0],
            wb[1][1] - wb[0][1],
            fill=False,
            ls=":",
            ec="0.5",
            lw=0.8,
            zorder=0,
        )
    )

    lk = dict(
        fontsize=8,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )
    if label_kwargs:
        lk.update(label_kwargs)

    xs0, ys0 = [], []
    for name, s in geo.items():
        c = _color(s, name)
        if s["type"] == "rect":
            x0, y0 = s["xy_min"]
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    s["xy_max"][0] - x0,
                    s["xy_max"][1] - y0,
                    facecolor=c,
                    alpha=alpha,
                    edgecolor=c,
                    zorder=2,
                )
            )
            if s["kind"] == "free" and show_spawn_outline:
                sx0, sy0 = s["spawn_xy_min"]
                ax.add_patch(
                    Rectangle(
                        (sx0, sy0),
                        s["spawn_xy_max"][0] - sx0,
                        s["spawn_xy_max"][1] - sy0,
                        fill=False,
                        edgecolor=c,
                        zorder=3,
                    )
                )
            cx, cy = 0.5 * (x0 + s["xy_max"][0]), 0.5 * (y0 + s["xy_max"][1])
            xs0 += [x0, s["xy_max"][0]]
            ys0 += [y0, s["xy_max"][1]]
        elif s["type"] == "arc":
            cx, cy = s["center"]
            r = s["radius"]
            th = np.linspace(s["theta0"], s["theta1"], 61)  # direction preserved
            arc = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
            ax.add_patch(
                Polygon(
                    np.vstack([[cx, cy], arc]),
                    facecolor=c,
                    alpha=alpha,
                    edgecolor=c,
                    zorder=2,
                )
            )
            mid = 0.5 * (s["theta0"] + s["theta1"])
            cx, cy = cx + 0.55 * r * np.cos(mid), cy + 0.55 * r * np.sin(mid)
            xs0 += [cx]
            ys0 += [cy]
        if label:
            ax.text(cx, cy, name, color="black", zorder=4, **lk)  # type: ignore

    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if xs0 and ys0:
        # dataLim must be updated with the extents (patches aren't seen by autoscale)
        ax.update_datalim(np.column_stack([xs0, ys0]))
        ax.autoscale_view()  # fit exactly to the geometry
        ax.margins(margin)  # then add margin so edge labels aren't cut
    return ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    args = parser.parse_args()

    for scene_cfg in iter_scene_configs():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        scene = Scene.get(scene_cfg)
        out_dir = Scene.save_dir(scene_cfg) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_scene_boundaries(scene, ax=ax)
        path = out_dir / f"floor_plan_{scene.cfg.tag}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[{scene_cfg.tag}] wrote floor plan to {path}")


if __name__ == "__main__":
    main()
