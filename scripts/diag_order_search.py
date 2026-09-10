"""Try every order of the goal-relevant fixes for one task (exhaustive over
orders, greedy within an order).

The virtual oracle teleports an option's ``target_entities`` straight to the
task goal, so a task is solvable iff the required per-entity fixes can be applied
in *some* order without the state gate blocking the next one.  These scenes'
pre-conditions encode the states of entities the skill does not move, so the
order matters: fixing entity A can invalidate the pre-condition of the option
that fixes entity B.

For each task this walks all permutations of the entities that differ from the
goal (``itertools.permutations``, 6! = 720 for scene9 *) and reports how many
orders solve the task.  Zero solved orders is strong evidence that the task is
unreachable for the given option vocabulary + gate, and the recorded blocking
step says which option/entity failed last.

Usage:
    python scripts/diag_order_search.py --scene scene9 --env-seed 12345 \
        --episodes 42,103,172,179,193 --out data/diag/orders_scene9.json
"""

import argparse
import itertools
import json
import sys
import time
from collections import Counter
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


def differing(x, y, tol) -> list[str]:
    out = []
    for label, _e in y.entities():
        try:
            vx = np.asarray(x.get(label).value, dtype=float)
            vy = np.asarray(y.get(label).value, dtype=float)
        except KeyError:
            continue
        if vx.shape == vy.shape and float(np.linalg.norm(vx - vy)) > tol:
            out.append(label)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--env-seed", type=int, default=12345)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--tol", type=float, default=1e-2)
    ap.add_argument("--max-perms", type=int, default=720)
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
    pairs = graph._option_conditions

    results = []
    for ep in eps:
        # same task for every permutation: reset the env RNG and sample again
        scene.env.np_random = np.random.default_rng(args.env_seed + ep)
        (x0, _), (y0, _) = scene.sample_task()
        graph.set_goal(y0)
        graph.set_start(x0)
        need = differing(x0, y0, args.tol)
        solved_orders, blocked_reasons = 0, Counter()
        first_fail = None
        n_orders = 0
        for perm in itertools.islice(itertools.permutations(need), args.max_perms):
            n_orders += 1
            scene.env.np_random = np.random.default_rng(args.env_seed + ep)
            (xx, _), (_yy, _) = scene.sample_task()
            graph.set_goal(y0)
            graph.set_start(xx)
            x = xx
            ok = False
            fail = None
            for ent in perm:
                if ent not in differing(x, y0, args.tol):
                    continue
                graph.set_start(x)
                try:
                    graph.export()
                    keys = graph.export_keys
                except RuntimeError:
                    fail = f"dead_end@{ent}"
                    break
                cands = [k for k in keys if ent in pairs[k].target_entities]
                if not cands:
                    fail = f"blocked@{ent}"
                    break
                idx = keys.index(cands[0])
                a, s = graph.select(idx)
                z, fb = ExpertModel.get(a).act(x, s)
                x = z
                if fb.terminal and fb.reward > 0.0:
                    ok = True
                    break
                if fb.truncated:
                    fail = f"truncated@{ent}"
                    break
            if ok:
                solved_orders += 1
            else:
                blocked_reasons[fail or "goal_not_reached"] += 1
                if first_fail is None:
                    first_fail = {"perm": list(perm), "fail": fail or "goal_not_reached",
                                  "remaining": differing(x, y0, args.tol)}
        rec = {"ep": ep, "n_entities": len(need), "entities": need,
               "orders_tried": n_orders, "orders_solved": solved_orders,
               "blocked_reasons": dict(blocked_reasons.most_common(6)),
               "first_failure": first_fail}
        results.append(rec)
        print(f"  [{args.scene}] ep {ep}: orders {solved_orders}/{n_orders} solved "
              f"({dict(blocked_reasons.most_common(3))})", flush=True)

    print(f"[{args.scene}] all-orders-solved: "
          f"{sum(1 for r in results if r['orders_solved'] > 0)}/{len(results)}")
    res = {"scene": args.scene, "env_seed": args.env_seed, "episodes": eps,
           "results": results, "elapsed_s": round(time.time() - t0, 1)}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
