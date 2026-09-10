"""Is a task's goal reachable at all by the option vocabulary?

Complement of ``diag_deadend_conditions.py``: that one asks "is the *start*
state inside some pre-condition"; this one asks "is the *goal* inside some
post-condition", i.e. can any option (or short chain of options) ever satisfy it.

For each (scene, episode) it reports

  * the goal delta ``x0 -> y0`` per entity (which entities have to change, and
    by how much), so the required level/state transitions are visible;
  * how many options *fully* accept the goal in their post-condition, and if that
    number is zero, per entity the best post-margin over all options that touch
    it, plus whether the goal value is inside that post-condition's training
    bounds (outside -> the goal needs a level no demonstration ever produced);
  * the same check after one committed option (depth-2 reachability), to separate
    "goal needs one skill we do not have" from "goal needs a specific order".

Episodes are reproducible: the env RNG is reset to ``default_rng(env_seed + ep)``
before every ``sample_task``.

Usage:
    python scripts/diag_goal_reachability.py --scene scene9 --env-seed 12345 \
        --episodes 42,103,172,179,193 --out data/diag/goal_reach_scene9.json
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.scenes.scene import Scene

from scripts.common.scenes import find_scene_config, find_scene_models
from scripts.diag_deadend_conditions import cond_value, fmt


def goal_acceptance(graph, y, x=None) -> dict:
    """Which options' post-conditions accept ``y`` from state ``x``?

    Returns counts plus, for entities that no option accepts, the best
    (smallest) margin seen and whether the goal value is inside the fitted
    post-condition bounds.
    """
    accepted, feasible = [], []
    per_entity = collections.defaultdict(list)
    for key in graph.ns_option.keys:
        node = graph.ns_option.get_by_key(key)
        con = graph._option_conditions.get(key)
        if con is None:
            continue
        checks = [cond_value(graph, l, con.post, "post", node) for l in con.post.models]
        for c in checks:
            if "z" in c:
                per_entity[c["label"]].append(c)
        if all(c.get("ok", False) for c in checks):
            accepted.append(key)
            if x is not None:
                feasible.append(key)
    best_per_entity = {}
    for label, checks in per_entity.items():
        best = min(checks, key=lambda c: c["z"])
        best_per_entity[label] = {
            "best_z": best["z"], "chi": best["chi"], "best_zd": best["zd_max"],
            "zd_cap": best["zd_cap"], "state_ok": best["state_ok"],
            "in_train_bounds": best["in_train_bounds"],
            "n_checks": len(checks),
        }
    return {"n_accepted": len(accepted), "accepted": accepted[:10],
            "per_entity_best": best_per_entity}


def goal_delta(x, y) -> dict:
    delta = {}
    for label, _e in y.entities():
        try:
            vx = np.asarray(x.get(label).value, dtype=float)
            vy = np.asarray(y.get(label).value, dtype=float)
        except KeyError:
            continue
        if vx.shape != vy.shape:
            continue
        d = np.abs(vx - vy)
        if d.max() > 1e-9:
            delta[label] = {"l2": round(float(np.linalg.norm(vx - vy)), 4),
                            "max_dim": int(np.argmax(d)),
                            "x": fmt(vx), "y": fmt(vy)}
    return delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--env-seed", type=int, default=12345)
    ap.add_argument("--episodes", required=True,
                    help="comma separated episode indices")
    ap.add_argument("--depth", type=int, default=2,
                    help="how many committed options to look ahead (1 = 1-step)")
    ap.add_argument("--branch", type=int, default=12,
                    help="max options expanded per level")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eps = [int(e) for e in args.episodes.split(",") if e.strip()]

    t0 = time.time()
    scene = Scene.get(find_scene_config(args.scene), auto_load=False)
    model_cfgs = find_scene_models(args.scene)
    for agent_cfg in model_cfgs:
        expert = ExpertModel.get(agent_cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
        _ = expert.conditions
    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode.BOTH)

    results = []
    for ep in eps:
        scene.env.np_random = np.random.default_rng(args.env_seed + ep)
        (x0, _), (y0, _) = scene.sample_task()
        graph.set_goal(y0)
        graph.set_start(x0)
        keys = graph.export_keys
        rec = {
            "ep": ep,
            "n_feasible_at_start": len(keys),
            "goal_delta": goal_delta(x0, y0),
            "goal_acceptance_from_start": goal_acceptance(graph, y0, x0),
        }
        # depth-2/…: commit one option, then ask again
        rec["depth_solved"] = []
        if args.depth > 1:
            snaps = {}
            depth_solved = rec["depth_solved"]
            for i, key in enumerate(keys[: args.branch]):
                graph.set_start(x0)
                a, s = graph.select(i)
                z, fb = ExpertModel.get(a).act(x0, s)
                if fb.terminal and fb.reward > 0.0:
                    depth_solved.append({"depth": 1, "via": key})
                    break
                graph.set_start(z)
                try:
                    keys2 = graph.feasible_keys()
                except RuntimeError:
                    continue
                acc = goal_acceptance(graph, y0)
                if acc["n_accepted"] > 0:
                    depth_solved.append(
                        {"depth": 2, "via": key, "then": acc["accepted"][:3]}
                    )
                snaps[key] = len(keys2)
            rec["depth_solved"] = depth_solved
            rec["second_step_options"] = snaps
        graph.set_start(x0)
        results.append(rec)
        print(f"  [{args.scene}] ep {ep}: feasible={rec['n_feasible_at_start']} "
              f"goal_accepted_by={rec['goal_acceptance_from_start']['n_accepted']} "
              f"depth_solved={len(rec['depth_solved'])}", flush=True)

    res = {"scene": args.scene, "env_seed": args.env_seed, "episodes": eps,
           "depth": args.depth, "results": results,
           "elapsed_s": round(time.time() - t0, 1)}
    print(f"[{args.scene}] episodes={len(eps)} "
          f"goal_unsupported={sum(1 for r in results if r['goal_acceptance_from_start']['n_accepted'] == 0)} "
          f"solved_within_depth={sum(1 for r in results if r['depth_solved'])}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
