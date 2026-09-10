"""Changeability closure: which entities can still be changed from this state?

Motivation: deciding "the goal is unreachable, end the episode" from the
*currently feasible* options is unsound.  An option that targets the missing
entity may be gated off only because some *other* entity sits in the wrong state
(e.g. a button in state 2 blocks a skill whose pre demos started at state 0) —
and that other entity can be fixed in a later step, which makes the option
feasible again.

Correct test: compute the set ``R`` of entities that can *ever* change, as the
least fixpoint of

    option o becomes applicable  iff  every pre-entity label l of o is
                                      (passes the gate right now)  or  (l ∈ R)
    if o is applicable -> every entity in ``o.target_entities`` is in R

The current gate status only *seeds* the iteration; options are not filtered by
feasibility.  Adding to ``R`` only ever enables more options (monotone), and
moving an entity can never be required to help itself, so ``R`` is an
*over*-approximation of the truly changeable entities.  Therefore

    goal-diff entity e with e ∉ R   =>   e can never be changed   =>   the goal is
                                         unreachable (sound, may miss cases)

The closure is deliberately optimistic, i.e. it cannot prove reachability: in
scene9 the mutually-blocking button/faucet pre-conditions make ``R`` contain
everything, so the closure correctly refuses to call those tasks dead.

Usage:
    python scripts/diag_changeability.py --scene scene4 --env-seed 12345 \
        --episodes 0,2,4 --out data/diag/changeability_scene4.json
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
from scripts.diag_deep_guided import mismatch


def pre_label_status(graph) -> dict:
    """Per (option, pre-entity) gate verdict at the graph's current start."""
    out = {}
    start_vals = graph._start_values()
    for key in graph.ns_option.keys:
        con = graph._option_conditions.get(key)
        if con is None:
            continue
        for label in con.pre.models:
            value = start_vals.get(label)
            if value is None:
                out[(key, label)] = False
                continue
            params = graph._prepared_params(con.pre.models[label], graph.entities[label])
            out[(key, label)] = bool(
                graph.entities[label].score_prepared(value, params)
            )
    return out


def targets_of(graph, key: str) -> set:
    con = graph._option_conditions.get(key)
    return set(con.target_entities) if con is not None else set()


def feasible_keys(graph) -> list:
    try:
        return graph.feasible_keys()
    except RuntimeError:
        return []


def changeable_closure(graph, status: dict) -> set:
    """Least fixpoint of the applicability rule (see module docstring)."""
    opts = {k: graph._option_conditions[k] for k in graph.ns_option.keys
            if k in graph._option_conditions}
    R: set = set()
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for key, con in opts.items():
            labels = list(con.pre.models)
            if all(status.get((key, l), False) or l in R for l in labels):
                t = set(con.target_entities)
                if not t <= R:
                    R |= t
                    changed = True
    return R


def naive_feasible_union(graph, keys: list) -> set:
    """Unsound shortcut: entities targeted by *currently feasible* options."""
    return {e for k in keys for e in targets_of(graph, k)}


def goal_diff(x, y, tol: float = 1e-2) -> list:
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


def verdicts(graph, x, y) -> dict:
    keys = feasible_keys(graph)
    status = pre_label_status(graph)
    R = changeable_closure(graph, status)
    naive = naive_feasible_union(graph, keys)
    diff = goal_diff(x, y)
    return {
        "n_options": len(graph.ns_option.keys),
        "n_feasible": len(keys),
        "goal_diff": diff,
        "changeable_R": sorted(R),
        "feasible_union": sorted(naive),
        "unreachable_naive": sorted(e for e in diff if e not in naive),
        "unreachable_closure": sorted(e for e in diff if e not in R),
        "empty_gate": len(keys) == 0,
        "mismatch": round(mismatch(x, y), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--env-seed", type=int, default=12345)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--max-steps", type=int, default=12)
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
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)
        rec = {"ep": ep, "at": "start", **verdicts(graph, x0, y0)}
        print(f"  [{args.scene}] ep {ep} start: feasible={rec['n_feasible']} "
              f"goal_diff={rec['goal_diff']} unreachable(naive)={rec['unreachable_naive']} "
              f"unreachable(closure)={rec['unreachable_closure']}", flush=True)
        results.append(rec)
    res = {"scene": args.scene, "env_seed": args.env_seed, "episodes": eps,
           "results": results, "elapsed_s": round(time.time() - t0, 1)}
    disagree = [r["ep"] for r in results
                if r["unreachable_naive"] != r["unreachable_closure"]]
    print(f"[{args.scene}] episodes where naive (feasible-only) and closure "
          f"disagree: {disagree}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
