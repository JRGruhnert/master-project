"""Targeted repair: fix one goal-relevant entity per step, and report the first
entity that *no feasible option* can touch.

The virtual oracle teleports the option's ``target_entities`` straight to the
task goal (``Scene._step_virt`` builds the subgoal from ``y``), so a task is
solvable by this oracle whenever, for every entity that differs from the goal,
*some* option that targets it is feasible. This script therefore walks the
remaining goal-delta entity by entity and stops at the first entity for which
either

  * no feasible option targets it ("BLOCKED"), or
  * the graph has no feasible option at all ("DEAD_END").

It then prints, for every blocked entity, the options that *would* touch it and
why each of them is rejected (failing pre-condition entity, margin, whether the
value is inside the fitted training bounds) - i.e. which condition forbids the
goal from being reached.

Usage:
    python scripts/diag_targeted_repair.py --scene scene9 --env-seed 12345 \
        --episodes 42,103,172,179,193 --out data/diag/repair_scene9.json
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
from scripts.diag_deadend_conditions import cond_value, fmt
from scripts.diag_deep_guided import mismatch
from scripts.diag_scene_reachability import env_restore, env_snapshot


def component_report(entity, value, up) -> list[dict]:
    """Per-mixture-component view of one gate check.

    ``Entity.score_single`` tests the *best pose component*: if that component's
    categorical state distribution does not contain the observed state, the
    check fails even when another component explains pose *and* state. This
    reports every component so that case becomes visible.
    """
    import math

    from scipy.stats import chi2

    sample = entity.model_value(value)
    p = entity.secure_mix_parameters(up, add_variance=True)
    pose, state = sample[:-1], int(sample[-1])
    weights = p["weights"]
    means = p["measurement"]["pose"]["means"]
    covs = p["measurement"]["pose"]["covariances"]
    pis = p["measurement"]["state"]["pis"]
    chi = float(math.sqrt(chi2.ppf(entity.cfg.z_quantile_joint, len(pose))))
    zd_cap = float(entity._z_dim_sigma)
    rows = []
    for k in range(len(weights)):
        var = np.maximum(covs[k], 1e-15)
        zd = np.abs(pose - means[k]) / np.sqrt(var)
        z = float(np.sqrt(np.sum(zd**2)))
        pose_ok = z <= chi and bool(np.all(zd <= zd_cap))
        p_state = float(pis[k][state])
        rows.append({"k": k, "weight": round(float(weights[k]), 4),
                     "z": round(z, 3), "zd_max": round(float(np.max(zd)), 3),
                     "pose_ok": bool(pose_ok), "p_state": p_state,
                     "state_ok": bool(p_state > 1e-6),
                     "accepts": bool(pose_ok and p_state > 1e-6)})
    return rows


def remaining_entities(x, y, tol: float) -> dict:
    """Entities whose value still differs from the goal (l2 > tol)."""
    out = {}
    for label, _e in y.entities():
        try:
            vx = np.asarray(x.get(label).value, dtype=float)
            vy = np.asarray(y.get(label).value, dtype=float)
        except KeyError:
            continue
        if vx.shape != vy.shape:
            continue
        d = float(np.linalg.norm(vx - vy))
        if d > tol:
            out[label] = round(d, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--env-seed", type=int, default=12345)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--tol", type=float, default=1e-2)
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
    max_steps = args.max_steps or scene.cfg.max_steps

    # option key -> ConPair (target_entities) for the "which entity does it touch" map
    pairs = graph._option_conditions

    def options_touching(label):
        return [k for k, con in pairs.items() if label in con.target_entities]

    results = []
    for ep in eps:
        scene.env.np_random = np.random.default_rng(args.env_seed + ep)
        (x0, _), (y0, _) = scene.sample_task()
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)
        x = x0
        trace = []
        status = "budget"
        blocked = {}
        for step in range(max_steps):
            rem = remaining_entities(x, y0, args.tol)
            if not rem:
                status = "solved_state"
                break
            try:
                graph.export()
                keys = graph.export_keys
            except RuntimeError:
                status = "dead_end"
                blocked = {"-": rem}
                break
            feasible = set(keys)
            cands = [k for k in keys
                     if set(pairs[k].target_entities) & set(rem)]
            if not cands:
                status = "blocked"
                # why can no option touch the remaining entities?
                for label in rem:
                    det = []
                    for k in options_touching(label):
                        node = graph.ns_option.get_by_key(k)
                        con = pairs[k]
                        checks = [cond_value(graph, l, con.pre, "pre", node)
                                  for l in con.pre.models]
                        # per-component view for the failing label itself
                        comp = {}
                        if label in con.pre.models:
                            up = con.pre.models[label].get_parameters().copy()
                            try:
                                val = graph.start.get(label).value
                                comp[label] = component_report(
                                    graph.entities[label], val, up)
                            except (KeyError, TypeError):
                                comp = {}
                        det.append({
                            "key": k,
                            "feasible": k in feasible,
                            "pre_fail": [c["label"] for c in checks
                                         if c.get("ok") is False],
                            "checks": checks,
                            "components": comp,
                        })
                    blocked[label] = det
                break
            # Entity with the largest gap first, then pick the candidate by its
            # *outcome*: a plain option teleports its target to the goal
            # (ValueMode.GOAL), but the "<label>s" SAMPLE variant teleports to a
            # fresh random draw from the post distribution, so picking it by name
            # order oscillates forever instead of converging.
            target = max(rem, key=lambda e: rem[e])
            cands_t = [k for k in cands
                       if target in pairs[k].target_entities] or cands
            snap = env_snapshot(scene)
            step_before = scene.current_step
            scored = []
            for k in cands_t:
                env_restore(scene, snap)
                scene.current_step = step_before
                graph.set_start(x)
                graph.export()
                here = graph.export_keys
                if k not in here:
                    continue
                a_k, s_k = graph.select(here.index(k))
                z_k, fb_k = ExpertModel.get(a_k).act(x, s_k)
                scored.append((mismatch(z_k, y0), k,
                               bool(fb_k.terminal and fb_k.reward > 0.0)))
            if not scored:
                status = "blocked"
                blocked = {target: []}
                break
            _, best, _ = min(scored, key=lambda t: t[0])
            env_restore(scene, snap)
            scene.current_step = step_before
            graph.set_start(x)
            graph.export()
            here = graph.export_keys
            idx = here.index(best)
            a, s = graph.select(idx)
            z, fb = ExpertModel.get(a).act(x, s)
            trace.append({
                "step": step, "option": best,
                "sample_variant": best.endswith("s"),
                "candidates_scored": len(scored),
                "targets": sorted(set(pairs[best].target_entities) & set(rem)),
                "remaining_before": rem,
                "reward": float(fb.reward), "terminal": bool(fb.terminal),
                "truncated": bool(fb.truncated),
            })
            x = z
            graph.set_start(x)
            if fb.terminal and fb.reward > 0.0:
                status = "solved_env"
                break
            if fb.truncated:
                status = "truncated"
                break
        rec = {"ep": ep, "status": status, "n_steps": len(trace),
               "remaining": remaining_entities(x, y0, args.tol),
               "trace": trace}
        # sound unreachability check at the state where the walk stopped
        from scripts.diag_changeability import verdicts

        graph.set_start(x)
        rec["unreachability"] = verdicts(graph, x, y0)
        if blocked:
            rec["blocked"] = {
                label: [
                    {"key": d["key"], "feasible": d["feasible"],
                     "pre_fail": d["pre_fail"],
                     "components": d.get("components", {}),
                     "checks": [
                         {kk: c[kk] for kk in
                          ("label", "ok", "z", "chi", "zd_max", "zd_cap",
                           "state_ok", "in_train_bounds") if kk in c}
                         for c in d["checks"]]}
                    for d in det]
                for label, det in blocked.items()
            }
        results.append(rec)
        print(f"  [{args.scene}] ep {ep}: {status} after {len(trace)} steps, "
              f"remaining={rec['remaining']}", flush=True)

    res = {"scene": args.scene, "env_seed": args.env_seed, "episodes": eps,
           "tol": args.tol, "results": results,
           "elapsed_s": round(time.time() - t0, 1)}
    from collections import Counter
    print(f"[{args.scene}] statuses: {dict(Counter(r['status'] for r in results))}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
