import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import cast

from heca.data.condition import Condition
from heca.data.entity import Entity
from heca.scenes.ogbench.scene import OGScene
from heca.scenes.scene import Scene

# Make ``conf`` / ``scripts.common`` importable when run directly as
# ``python scripts/evaluate_tapas.py`` (mirrors scripts/__init__.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt
import numpy as np
import torch
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase
from heca.data.data import DCEntity, DCScene
from heca.experts.expert import ExpertModel
from heca.misc import logger
from scripts.common.args import (
    add_model_argument,
    add_scene_argument,
    add_tag_argument,
    add_use_gt_argument,
    add_viewer_argument,
)
from scripts.common.plot_lock import PLOT_LOCK
from scripts.common.scenes import agents_by_scene


def sample_dcscene(con: Condition) -> DCScene:
    """Sample one value per entity from a condition's fitted models."""
    dc: dict[str, DCEntity] = {}
    for label, entity in con.entities.items():
        value = con.models[label].sample(1)[0]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value).squeeze()
        value = entity.model_to_value(value)
        dc[label] = DCEntity(value=value, feature=entity.gnn_format(value))
    return DCScene(dc)


def condition_info_dict(con: Condition, scene: Scene) -> dict:
    dcscene = sample_dcscene(con)
    info: dict = {}
    for label, entity in con.entities.items():
        info.update(
            entity.env_state_value(
                label, dcscene, unnormalize_pos=scene.unnormalize_position
            )
        )
    return info


def anchor_entities(agent: ExpertModel) -> set[str]:
    return {
        label
        for label, score in agent.conditions.change_scores.items()
        if score < Entity.ANCHOR_THRESHOLD
    }


def sample_task_conditions(
    agent: ExpertModel, scene: Scene, anchors: set[str]
) -> tuple[dict, dict]:
    pair = agent.conditions
    pre_info = condition_info_dict(pair.pre, scene)
    post_info = condition_info_dict(pair.post, scene)
    for label in anchors:
        for key in list(post_info):
            if key.startswith(f"heca_{label}_"):
                post_info[key] = pre_info[key]
    return pre_info, post_info


def get_env_safely(scene: Scene) -> SceneEnvBase:
    assert isinstance(scene, OGScene), "Only OgScene Supported."
    return cast(SceneEnvBase, scene.env.unwrapped)


def evaluate_model(
    model: ExpertModel,
    scene: Scene,
    episodes: int,
    max_tries: int,
) -> Counter:
    env = get_env_safely(scene)
    counts: Counter = Counter()
    anchors = anchor_entities(model)
    for ep in range(episodes):
        env.reset(options={"render_goal": True})
        pre_info, post_info = sample_task_conditions(model, scene, anchors)

        env.set_start(pre_info)
        env.set_goal(post_info)
        y = scene.to_dc_scene(env.get_reset_info()["goal"])

        succeeded_on = 0
        for attempt in range(1, max_tries + 1):
            x = scene.to_dc_scene(env.compute_ob_info())
            _, fb = model.act(x, y)
            if fb.success:
                succeeded_on = attempt
                break

        counts[succeeded_on] += 1
        logger.debug(
            f"[{scene.cfg.tag}] {model.cfg.tag} episode {ep + 1}/{episodes}: "
            f"{'success on try ' + str(succeeded_on) if succeeded_on else 'failed'}"
        )
    return counts


def _plot_scene_impl(
    scene_tag: str,
    results: list[dict],
    out_dir: Path,
    max_tries: int,
    episodes: int,
) -> Path:
    results = [
        {**r, "counts": {int(k): v for k, v in r["counts"].items()}} for r in results
    ]
    tags = [r["tag"] for r in results]
    n = len(tags)
    pct = np.zeros((n, max_tries + 1))
    for i, r in enumerate(results):
        c = r["counts"]
        total = sum(c.values()) or 1
        for t in range(1, max_tries + 1):
            pct[i, t - 1] = 100.0 * c.get(t, 0) / total
        pct[i, max_tries] = 100.0 * c.get(0, 0) / total  # failed / gave up

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * n), 6.0))
    xpos = np.arange(n)
    bottom = np.zeros(n)
    cmap = plt.get_cmap("summer")
    colors = [cmap(k / max(1, max_tries - 1)) for k in range(max_tries)]
    for t in range(max_tries):
        ax.bar(
            xpos,
            pct[:, t],
            bottom=bottom,
            width=0.65,
            label=f"{t + 1}. try",
            color=colors[t],
        )
        bottom += pct[:, t]
    ax.bar(
        xpos,
        pct[:, max_tries],
        bottom=bottom,
        width=0.65,
        label="failed",
        color="1.0",
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("episodes [%]")
    ax.set_title(f"Tapas success reliability — {scene_tag} ({episodes} episodes/agent)")
    ax.legend(loc="upper right", fontsize=8)

    # Total success rate above each bar (success on any attempt).
    for i, r in enumerate(results):
        c = r["counts"]
        total = sum(c.values())
        ok = total - c.get(0, 0)
        ax.text(
            i,
            101.5,
            f"{100.0 * ok / total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    path = out_dir / f"eval_success_{scene_tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scene(
    scene_tag: str,
    results: list[dict],
    out_dir: Path,
    max_tries: int,
    episodes: int,
) -> Path:
    """Stacked bar chart of per-attempt success rates (pyplot lock held)."""
    with PLOT_LOCK:
        return _plot_scene_impl(scene_tag, results, out_dir, max_tries, episodes)


def results_path(scene_cfg: Scene.Config) -> Path:
    """Shared, stateless results file for one scene."""
    out_dir = Scene.save_dir(scene_cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"eval_success_{scene_cfg.tag}.json"


def load_results(json_path: Path) -> dict:
    if json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def update_results(json_path: Path, entry: dict, episodes: int, max_tries: int):
    """Insert/replace one agent's entry in the scene results json (in place).

    Stateless: reads the current file, updates only this model's record, writes
    back. Safe to call per model so partial/interrupted runs keep their state.
    """
    data = load_results(json_path)
    data["episodes"] = episodes
    data["max_tries"] = max_tries
    agents = data.setdefault("agents", [])
    for i, agent in enumerate(agents):
        if agent.get("tag") == entry["tag"]:
            agents[i] = entry
            break
    else:
        agents.append(entry)
    data.setdefault("failures", [])
    json_path.write_text(json.dumps(data, indent=2))
    return data


def plot_results(scene_cfg: Scene.Config, json_path: Path | None = None) -> Path | None:
    """(Re)plot the current scene chart from the persisted results json."""
    json_path = json_path or results_path(scene_cfg)
    data = load_results(json_path)
    agents = data.get("agents", [])
    if not agents:
        return None
    episodes = int(data.get("episodes", 100))
    max_tries = int(data.get("max_tries", 3))
    return plot_scene(scene_cfg.tag, agents, json_path.parent, max_tries, episodes)


def evaluate_one(
    cfg: ExpertModel.Config,
    scene_cfg: Scene.Config,
    episodes: int,
    max_tries: int,
    gt: bool,
) -> Counter:
    """Evaluate one model, persist its result, re-plot the scene chart."""
    model = ExpertModel.get(cfg).use_gt(gt)
    counts = evaluate_model(model, model.scene, episodes, max_tries)

    json_path = results_path(scene_cfg)
    update_results(
        json_path,
        {"scene": scene_cfg.tag, "tag": cfg.tag, "counts": dict(counts)},
        episodes,
        max_tries,
    )
    plot_path = plot_results(scene_cfg, json_path)

    total = sum(counts.values())
    ok = total - counts.get(0, 0)
    hist = ", ".join(f"try{t}={counts.get(t, 0)}" for t in range(1, max_tries + 1))
    logger.info(
        f"[{scene_cfg.tag}] {cfg.tag}: {ok}/{total} episodes ok "
        f"({hist}, failed={counts.get(0, 0)})"
    )
    if plot_path is not None:
        logger.info(f"[{scene_cfg.tag}] updated {plot_path}")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_viewer_argument(parser)
    add_use_gt_argument(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Max attempts per episode before giving up.",
    )

    args = parser.parse_args()

    for scene_cfg, models in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        if args.viewer:
            Scene.get(scene_cfg, auto_load=False).cfg.viewer = True
        logger.info(f"[{scene_cfg.tag}] evaluating {len(models)} agents")

        for cfg in models:
            if args.model and cfg.tag != args.model:
                continue
            evaluate_one(cfg, scene_cfg, args.episodes, args.max_tries, args.gt)


if __name__ == "__main__":
    main()
