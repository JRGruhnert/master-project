"""Scene-generic oracle reachability/solvability test (scenes 0..10).

For each sampled task, run random option chains with full env isolation
(snapshot/restore) and up to `--tries` retries per episode; an episode counts
as reachable if some chain reaches fb.terminal with reward > 0 within the
scene's option budget (Scene.count_option / cfg.max_steps).

Conditions are fitted on demand when a cache is missing (the refactor moved
heca.conditions.* -> heca.data.*, invalidating old joblib pickles).

Usage:
    python scripts/diag_scene_reachability.py --scene scene1 \
        --episodes 60 --tries 80 --out data/diag/reach_scene1.json
"""

import argparse
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


def env_snapshot(scene) -> tuple:
    env = scene.env
    obj_recs = []
    for o in env.objects:
        rec = {}
        for attr in ("_cur_state", "_target_button_states"):
            v = getattr(o, attr, None)
            rec[attr] = v.copy() if v is not None else None
        rec["_target_val"] = getattr(o, "_target_val", None)
        obj_recs.append((o, rec))
    return env._data.qpos.copy(), env._data.qvel.copy(), obj_recs


def env_restore(scene, snap) -> None:
    env = scene.env
    qpos, qvel, obj_recs = snap
    env._data.qpos[:] = qpos
    env._data.qvel[:] = qvel
    for o, rec in obj_recs:
        for attr, val in rec.items():
            if val is None:
                continue
            if attr == "_target_val":
                setattr(o, attr, val)
            else:
                cur = getattr(o, attr)
                cur[:] = val
    env._apply_button_states()
    env._success = bool(env._evaluate_success(env._compute_successes()))


def try_export(graph):
    """graph.export() but tolerate an empty feasible set.

    ``Graph.feasible_keys`` raises RuntimeError when no option passes the state
    gate (the old full-set fallback was removed). Returns (keys, None) on
    success or (None, "gate") when nothing is feasible.
    """
    try:
        graph.export()
        return graph.export_keys, None
    except RuntimeError:
        return None, "gate"


def run_chain(x, y, graph, pick, max_opts):
    """One chain from x (env already at x).

    Returns (ok, n_opts, dead_end) where dead_end is "gate" if the chain hit a
    state with no feasible option at all.
    """
    n = 0
    for _ in range(max_opts):
        keys, dead = try_export(graph)
        if keys is None:
            return False, n, dead
        i = pick(len(keys))
        a, s = graph.select(i)
        z, fb = ExpertModel.get(a).act(x, s)
        n += 1
        x = z
        graph.set_start(x)
        if fb.terminal:
            return fb.reward > 0.0, n, None
        if fb.truncated:
            return False, n, None
    return False, n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--tries", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smode", type=str, default="both",
                    choices=["none", "simple", "chain", "both"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    scene_cfg = find_scene_config(args.scene)
    model_cfgs = find_scene_models(args.scene)
    scene = Scene.get(scene_cfg, auto_load=False)

    print(f"[{args.scene}] loading {len(model_cfgs)} experts "
          f"(conditions fit on demand if cache missing)…", flush=True)
    for i, agent_cfg in enumerate(model_cfgs):
        t = time.time()
        expert = ExpertModel.get(agent_cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
        _ = expert.conditions  # triggers fit / cache load
        print(f"  [{i+1}/{len(model_cfgs)}] {agent_cfg.tag:<22} "
              f"conditions ready in {time.time()-t:5.1f}s", flush=True)

    smode = SubgoalMode(args.smode)
    graph = Graph.generate(list(model_cfgs), smode=smode)
    n_options = len(graph.ns_option.keys)
    print(f"[{args.scene}] graph built: {n_options} options "
          f"(smode={smode.name}) after {time.time()-t0:.1f}s", flush=True)

    max_opts = scene.cfg.max_steps
    rng = np.random.RandomState(args.seed)
    n_solv = 0
    lens = []
    enabled_sizes = []
    n_dead_end_start = 0
    n_dead_end_any = 0
    per_ep = []

    for ep in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()
        snap0 = env_snapshot(scene)
        scene.current_step = 0

        graph.set_goal(y0)
        graph.set_start(x0)
        keys0, dead0 = try_export(graph)
        if keys0 is None:
            n_dead_end_start += 1
            enabled_sizes.append(0)
        else:
            enabled_sizes.append(len(keys0))

        solved = False
        best_len = None
        hit_dead_end = dead0 is not None
        if keys0 is not None:
            for _ in range(args.tries):
                env_restore(scene, snap0)
                scene.current_step = 0
                graph.set_goal(y0)
                graph.set_start(x0)
                ok, ln, dead = run_chain(
                    x0, y0, graph, lambda n: int(rng.randint(n)), max_opts
                )
                if dead is not None:
                    hit_dead_end = True
                if ok:
                    solved = True
                    lens.append(ln)
                    best_len = ln
                    break
        if hit_dead_end:
            n_dead_end_any += 1
        n_solv += int(solved)
        per_ep.append(
            {"ep": ep, "solved": solved, "len": best_len, "dead_end": hit_dead_end}
        )
        if (ep + 1) % 10 == 0:
            print(f"  [{args.scene}] {ep+1}/{args.episodes} episodes "
                  f"({n_solv} solvable so far, {n_dead_end_any} dead-ends)",
                  flush=True)

    res = {
        "scene": args.scene,
        "smode": smode.name,
        "episodes": args.episodes,
        "tries": args.tries,
        "seed": args.seed,
        "n_options": n_options,
        "n_actionable_experts": len(model_cfgs),
        "solvable": n_solv,
        "solvable_pct": 100.0 * n_solv / max(args.episodes, 1),
        "mean_oracle_chain_len": float(np.mean(lens)) if lens else None,
        "max_oracle_chain_len": int(max(lens)) if lens else None,
        "max_opts": max_opts,
        "enabled_at_start_mean": float(np.mean(enabled_sizes)) if enabled_sizes else None,
        "enabled_at_start_min": int(np.min(enabled_sizes)) if enabled_sizes else None,
        "episodes_with_zero_enabled": n_dead_end_start,
        "episodes_hitting_dead_end": n_dead_end_any,
        "elapsed_s": round(time.time() - t0, 1),
        "per_episode": per_ep,
    }
    print(f"[{args.scene}] RESULT: solvable {n_solv}/{args.episodes} = "
          f"{res['solvable_pct']:.1f}%  "
          f"mean_chain_len={res['mean_oracle_chain_len']}  "
          f"enabled@start mean={res['enabled_at_start_mean']:.1f} "
          f"min={res['enabled_at_start_min']}  "
          f"({res['elapsed_s']}s)", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"[{args.scene}] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
