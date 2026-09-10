"""Name the *conditions* that make ``Graph.feasible_keys`` raise (empty gate).

``diag_deadends.py`` shows *that* every option fails in a dead-end state; this
script answers *which condition* is responsible and *whether a refit could
help*:

  * per option, per entity: the gate margin of the failing pre/post condition
    (chi2 radius ``z`` vs cap, worst per-dim sigma ``zd`` vs cap, state-logit
    check) so a marginal miss (candidate for re-fitting the condition) is
    distinguishable from a state the condition never saw;
  * whether the offending value lies inside the fitted training bounds
    (``ConPair.pre.data_bounds``) - outside means no re-fit on the same data
    can ever accept it, only new data / a new skill can;
  * the level signature of the state relative to the *episode's own* anchors
    (cube in box, cube on shelf, lid on box = box closed, entity off base);
  * a single-entity intervention: restore one entity to a sample of the
    base-condition that wants it and re-run the gate.  If the gate opens, that
    entity is the blocker and the option that opens is the missing link.

Usage:
    python scripts/diag_deadend_conditions.py --scene scene4 --episodes 10 \
        --tries 20 --max-deadends 10 --out data/diag/deadend_cond_scene4.json
"""

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.stats import chi2

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.scenes.scene import Scene

from scripts.common.scenes import find_scene_config, find_scene_models
from scripts.diag_scene_reachability import env_restore, env_snapshot

# movable entity -> (anchor, xy radius, dz window) used for the level signature
BOX_XY = 0.30
SHELF_XY = 0.45


def fmt(v):
    return [round(float(x), 3) for x in np.asarray(v).ravel()]


def margin(entity, value, up) -> dict:
    """Gate margin of one fitted condition for one value."""
    sample = entity.model_value(value)
    p = entity.secure_mix_parameters(up, add_variance=True)
    pose = sample[:-1]
    state = int(sample[-1])
    best_k, z, zd = entity._best_component(pose, p)
    pis = p["measurement"]["state"]["pis"]
    return {
        "chi": round(float(math.sqrt(chi2.ppf(entity.cfg.z_quantile_joint, len(pose)))), 3),
        "z": round(float(z), 3),
        "zd_cap": round(float(entity._z_dim_sigma), 3),
        "zd_max": round(float(np.max(zd)), 3),
        "state_ok": bool(pis[best_k][state] > 1e-6),
    }


def cond_value(graph, label, con, kind, node) -> dict:
    """Evaluate one entity of a pre/post condition and report why it fails."""
    up = con.models[label].get_parameters().copy()
    src = graph.start if kind == "pre" else graph.assemble_subgoal(node)
    try:
        value = src.get(label).value
    except (KeyError, TypeError):
        return {"label": label, "kind": kind, "missing_key": True}
    ent = graph.entities[label]
    m = margin(ent, value, up)
    bounds = con.data_bounds.get(label)
    inside = None
    if bounds is not None:
        lo, hi = bounds
        inside = bool(np.all(np.asarray(value) >= lo) and np.all(np.asarray(value) <= hi))
    ok = m["z"] <= m["chi"] and m["zd_max"] <= m["zd_cap"] and m["state_ok"]
    return {"label": label, "kind": kind, "ok": bool(ok), "in_train_bounds": inside,
            **m}


def gate_report(graph, y) -> dict:
    """Full per-option gate report for the current start state."""
    per_option = {}
    pre_fail = collections.Counter()
    post_fail = collections.Counter()
    for key in graph.ns_option.keys:
        node = graph.ns_option.get_by_key(key)
        con = graph._option_conditions.get(key)
        if con is None:
            continue
        recs = [cond_value(graph, l, con.pre, "pre", node) for l in con.pre.models]
        p = graph.assemble_subgoal(node)
        recs += [cond_value(graph, l, con.post, "post", node) for l in con.post.models]
        # would the goal be accepted directly by this option's post-condition?
        goal_ok = all(
            cond_value(graph, l, con.post, "post", node)["ok"]
            for l in con.post.models
        ) if p is not None else False
        ok = all(r.get("ok", False) for r in recs)
        per_option[key] = {"ok": bool(ok), "goal_accepted": bool(goal_ok), "checks": recs}
        for r in recs:
            if not r.get("ok", False):
                (pre_fail if r["kind"] == "pre" else post_fail)[r["label"]] += 1
    return {
        "n_options": len(graph.ns_option.keys),
        "n_feasible": sum(v["ok"] for v in per_option.values()),
        "n_goal_accepted_by_post": sum(v["goal_accepted"] for v in per_option.values()),
        "pre_fail_by_entity": dict(pre_fail.most_common()),
        "post_fail_by_entity": dict(post_fail.most_common()),
        "per_option": per_option,
    }


def level_signature(x) -> dict:
    """Where is each movable entity, relative to this episode's own anchors?"""
    pos = {l: np.asarray(v.value).ravel()[:3] for l, v in x.entities()}
    anchor = None
    for name in ("box0",):
        if name in pos:
            anchor = pos[name]
    shelf = pos.get("shelf0")
    sig = {}
    for label, p in pos.items():
        if label in ("box0", "shelf0"):
            continue
        d = {}
        if anchor is not None:
            d_xy = float(np.linalg.norm(p[:2] - anchor[:2]))
            dz = float(p[2] - anchor[2])
            if d_xy < BOX_XY and dz > 0.45:
                d["level"] = "on_box"
            elif d_xy < BOX_XY and dz < 0.40:
                d["level"] = "in_box"
            d["d_box_xy"] = round(d_xy, 3)
            d["dz_box"] = round(dz, 3)
        if shelf is not None:
            d_shelf = float(np.linalg.norm(p[:2] - shelf[:2]))
            d["d_shelf_xy"] = round(d_shelf, 3)
            if d_shelf < SHELF_XY and float(p[2] - shelf[2]) > 0.5:
                prev = d.get("level")
                d["level"] = f"{prev or 'free'}|on_shelf"
        d.setdefault("level", "free/base")
        d["pos"] = fmt(p)
        sig[label] = d
    return sig


def intervention(graph, x, y, pre_labels) -> dict:
    """Restore one entity at a time to a sample of the base condition that
    wants it; does the gate open again?"""
    out = {}
    for label in sorted(pre_labels):
        for key in graph.ns_option.keys:
            con = graph._option_conditions.get(key)
            if con is None or label not in con.pre.models:
                continue
            try:
                base = con.pre.sample(label)
            except Exception as exc:  # noqa: BLE001 - diagnostic
                out.setdefault(label, []).append(
                    {"via": key, "error": f"{type(exc).__name__}: {exc}"})
                continue
            x2 = x.copy()
            x2.set(label, base)
            graph.set_start(x2)
            try:
                keys = graph.export_keys
                graph.export()
                opened = True
                n = len(graph.feasible_keys())
            except RuntimeError:
                keys, n, opened = [], 0, False
            graph.set_goal(y)
            graph.set_start(x)
            out.setdefault(label, []).append(
                {"via": key, "opened": bool(opened), "n_feasible": int(n),
                 "keys": keys[:5], "sampled_value": fmt(base.value)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene4")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-deadends", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
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
    max_opts = args.max_steps or scene.cfg.max_steps
    rng = np.random.RandomState(args.seed)

    pre_labels = set()
    for key in graph.ns_option.keys:
        con = graph._option_conditions.get(key)
        if con is not None:
            pre_labels.update(con.pre.models)

    deadends = []
    n_eps_with_deadend = 0
    agg_pre = collections.Counter()
    agg_post = collections.Counter()
    agg_blocker = collections.Counter()
    agg_bounds = collections.Counter()
    level_hist = collections.Counter()

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
                        rep = gate_report(graph, y0)
                        sig = level_signature(x)
                        inter = intervention(graph, x, y0, pre_labels)
                        blocking = sorted(
                            l for l, v in inter.items()
                            if any(r.get("opened") for r in v)
                        )
                        for l in blocking:
                            agg_blocker[l] += 1
                        for l, d in sig.items():
                            level_hist[(l, d["level"])] += 1
                        for opt in rep["per_option"].values():
                            for c in opt["checks"]:
                                if c.get("ok") is False:
                                    agg_bounds[
                                        (c["kind"], c["label"],
                                         "inside" if c["in_train_bounds"] else "outside",
                                         c["in_train_bounds"] is None)
                                    ] += 1
                        deadends.append({
                            "ep": ep, "step": step,
                            "report": rep,
                            "level_signature": sig,
                            "blocking_entity": blocking,
                            "intervention": inter,
                            "x": {l: fmt(v.value) for l, v in x.entities()},
                            "y": {l: fmt(v.value) for l, v in y0.entities()},
                        })
                        agg_pre.update(rep["pre_fail_by_entity"])
                        agg_post.update(rep["post_fail_by_entity"])
                    ep_has_deadend = True
                    break
                keys = graph.export_keys
                a, s = graph.select(int(rng.randint(len(keys))))
                z, fb = ExpertModel.get(a).act(x, s)
                x = z
                graph.set_start(x)
                if fb.terminal or fb.truncated:
                    break
            if ep_has_deadend or len(deadends) >= args.max_deadends:
                break
        if ep_has_deadend:
            n_eps_with_deadend += 1
        print(f"  [{args.scene}] ep {ep+1}/{args.episodes} deadend-eps="
              f"{n_eps_with_deadend} recorded={len(deadends)}", flush=True)
        if len(deadends) >= args.max_deadends:
            break

    res = {
        "scene": args.scene, "episodes": args.episodes, "tries": args.tries,
        "seed": args.seed, "max_steps": max_opts,
        "n_options": len(graph.ns_option.keys),
        "episodes_with_deadend": n_eps_with_deadend,
        "deadends_recorded": len(deadends),
        "pre_fail_by_entity": dict(agg_pre.most_common()),
        "post_fail_by_entity": dict(agg_post.most_common()),
        "blocking_entity_hist": dict(agg_blocker.most_common()),
        "failing_value_vs_train_bounds": {
            f"{k[0]}:{k[1]}:{'no_bounds' if k[3] else k[2]}": v
            for k, v in agg_bounds.most_common()
        },
        "level_hist": {f"{k[0]}@{k[1]}": v for k, v in level_hist.most_common()},
        "deadends": deadends,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[{args.scene}] dead-ends={len(deadends)} "
          f"eps_with_deadend={n_eps_with_deadend}")
    print(f"  pre-fail  by entity: {dict(agg_pre.most_common(8))}")
    print(f"  post-fail by entity: {dict(agg_post.most_common(8))}")
    print(f"  entity whose restore re-opens the gate: {dict(agg_blocker.most_common(8))}")
    print(f"  failing value vs training bounds: "
          f"{ {f'{k[0]}:{k[1]}:{k[2]}': v for k, v in agg_bounds.most_common(12)} }")
    print(f"  level histogram: { {f'{k[0]}@{k[1]}': v for k, v in level_hist.most_common(12)} }")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
