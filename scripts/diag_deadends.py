"""Diagnose empty-gate ("dead-end") states: states where NO option is feasible,
so ``Graph.feasible_keys`` raises RuntimeError.

Replays the same episode stream as ``diag_scene_reachability.py`` (same scene,
seed, episodes, tries), and whenever a chain reaches a state with no feasible
option it records why each option failed:

  * which options fail, and on which entity labels
  * pre-condition failures vs post-condition failures vs missing-key errors
  * whether the goal itself would be accepted by any option's post-condition
    (i.e. does a 1-step solution exist in the gate's view at all)

Usage:
    python scripts/diag_deadends.py --scene scene7 --episodes 60 --tries 60 \
        --max-deadends 40 --out data/diag/deadends_scene7.json
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
from scripts.diag_scene_reachability import env_restore, env_snapshot


def fmt(v, n=3):
    return [round(float(x), 2) for x in np.asarray(v).ravel()[:n]]


def classify(graph, x, y) -> dict:
    """Why does every option fail the gate in this state?"""
    pre_fail = collections.Counter()
    post_fail = collections.Counter()
    key_err = collections.Counter()
    option_fail = collections.Counter()
    per_option = []
    goal_accepted_by_post = 0

    for key in graph.ns_option.keys:
        node = graph.ns_option.get_by_key(key)
        con = graph._option_conditions.get(key)
        if con is None:
            continue
        pf, pof, kerr = [], [], []
        pre_ok = True
        for label in con.pre.models:
            up = con.pre.models[label].get_parameters().copy()
            try:
                value = graph.start.get(label).value
            except KeyError:
                kerr.append(("pre", label))
                pre_ok = False
                continue
            if not graph.entities[label].score_single(value, up):
                pf.append(label)
                pre_ok = False
        post_ok = True
        try:
            subgoal = graph.assemble_subgoal(node)
        except KeyError:
            post_ok = False
            kerr.append(("subgoal", "-"))
            subgoal = None
        if subgoal is not None:
            for label in con.post.models:
                up = con.post.models[label].get_parameters().copy()
                try:
                    sv = subgoal.get(label).value
                except KeyError:
                    kerr.append(("post", label))
                    post_ok = False
                    continue
                if not graph.entities[label].score_single(sv, up):
                    pof.append(label)
                    post_ok = False
        # does the GOAL fit this option's post-condition? (1-step reachability)
        if not post_ok:
            pass
        else:
            goal_ok = True
            for label in con.post.models:
                up = con.post.models[label].get_parameters().copy()
                try:
                    gv = y.get(label).value
                except KeyError:
                    goal_ok = False
                    break
                if not graph.entities[label].score_single(gv, up):
                    goal_ok = False
                    break
            if goal_ok:
                goal_accepted_by_post += 1

        if not (pre_ok and post_ok):
            for l in pf:
                pre_fail[l] += 1
            for l in pof:
                post_fail[l] += 1
            for kind, l in kerr:
                key_err[f"{kind}:{l}"] += 1
            option_fail[key] += 1
            if len(per_option) < 8:
                per_option.append(
                    {"key": key, "pre_fail": sorted(set(pf)),
                     "post_fail": sorted(set(pof)), "key_err": kerr}
                )
    return {
        "n_options": len(graph.ns_option.keys),
        "n_failing_options": len(option_fail),
        "pre_fail_by_entity": dict(pre_fail.most_common()),
        "post_fail_by_entity": dict(post_fail.most_common()),
        "key_errors": dict(key_err.most_common()),
        "options_whose_post_accepts_goal": goal_accepted_by_post,
        "examples": per_option,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-deadends", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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
    max_opts = scene.cfg.max_steps
    rng = np.random.RandomState(args.seed)

    deadends = []
    n_eps_with_deadend = 0
    agg_pre = collections.Counter()
    agg_post = collections.Counter()
    agg_keyerr = collections.Counter()
    agg_goal_accept = collections.Counter()

    for ep in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()
        snap0 = env_snapshot(scene)
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)

        ep_has_deadend = False
        for _ in range(args.tries):
            env_restore(scene, snap0)
            scene.current_step = 0
            graph.set_goal(y0)
            graph.set_start(x0)
            x = x0
            for step in range(max_opts):
                try:
                    graph.export()
                except RuntimeError:
                    if len(deadends) < args.max_deadends:
                        info = classify(graph, x, y0)
                        info.update(
                            {"ep": ep, "step": step,
                             "x": {l: fmt(x.get(l).value)
                                   for l, _ in x.entities()},
                             "y": {l: fmt(y0.get(l).value)
                                   for l, _ in y0.entities()}}
                        )
                        deadends.append(info)
                        agg_pre.update(info["pre_fail_by_entity"])
                        agg_post.update(info["post_fail_by_entity"])
                        agg_keyerr.update(info["key_errors"])
                        agg_goal_accept[info["options_whose_post_accepts_goal"]] += 1
                    ep_has_deadend = True
                    break
                keys = graph.export_keys
                a, s = graph.select(int(rng.randint(len(keys))))
                z, fb = ExpertModel.get(a).act(x, s)
                x = z
                graph.set_start(x)
                if fb.terminal or fb.truncated:
                    break
            if ep_has_deadend:
                break
        if ep_has_deadend:
            n_eps_with_deadend += 1
        if (ep + 1) % 10 == 0:
            print(f"  [{args.scene}] {ep+1}/{args.episodes} eps, "
                  f"deadend-eps={n_eps_with_deadend}, recorded={len(deadends)}",
                  flush=True)

    res = {
        "scene": args.scene,
        "episodes": args.episodes,
        "tries": args.tries,
        "seed": args.seed,
        "n_options": len(graph.ns_option.keys),
        "episodes_with_deadend": n_eps_with_deadend,
        "deadends_recorded": len(deadends),
        "aggregate_pre_fail_by_entity": dict(agg_pre.most_common()),
        "aggregate_post_fail_by_entity": dict(agg_post.most_common()),
        "aggregate_key_errors": dict(agg_keyerr.most_common()),
        "goal_accept_count_hist": dict(agg_goal_accept),
        "deadends": deadends,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[{args.scene}] episodes_with_deadend={n_eps_with_deadend} "
          f"recorded={len(deadends)}")
    print(f"  pre-fail by entity : {dict(agg_pre.most_common(10))}")
    print(f"  post-fail by entity: {dict(agg_post.most_common(10))}")
    print(f"  key errors         : {dict(agg_keyerr.most_common(10))}")
    print(f"  #options whose post accepts the goal (per dead-end state): "
          f"{dict(agg_goal_accept)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
