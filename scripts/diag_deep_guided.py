"""Guided oracle for episodes that random search could not solve.

Replays the same episode stream as ``diag_scene_reachability.py`` (same scene,
seed, episodes, tries), keeps the episodes the random oracle failed on, then
attacks them with a *guided* solver:

  greedy   : at every state, try each feasible option in isolation
             (env snapshot/restore), keep the one that most reduces the
             discrepancy to the goal, commit it, repeat. Restarts with
             randomized tie-breaking to escape local optima.
  lookahead: like greedy, but for the best K candidates also evaluate the best
             follow-up option (2-step) and prefer the best pair.

This cannot prove unsolvability, but if a guided search with many restarts also
fails while random search fails, the goal is a strong candidate for a genuine
coverage hole.

Usage:
    python scripts/diag_deep_guided.py --scene scene9 --episodes 60 --tries 60 \
        --restarts 8 --out data/diag/guided_scene9.json
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
from scripts.diag_scene_reachability import env_restore, env_snapshot, run_chain, try_export


def mismatch(x, y) -> float:
    """Generic scene discrepancy: summed normalized per-entity distance."""
    total = 0.0
    n = 0
    for label, _ent in y.entities():
        try:
            vx = np.asarray(x.get(label).value, dtype=float)
            vy = np.asarray(y.get(label).value, dtype=float)
        except KeyError:
            continue
        if vx.shape != vy.shape:
            continue
        d = float(np.linalg.norm(vx - vy))
        total += d
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--episode-list", default=None,
                    help="comma separated episode indices to run instead of "
                         "range(--episodes); use together with --env-seed to "
                         "revisit specific reproducible tasks")
    ap.add_argument("--loose-gate", action="store_true",
                    help="patch Entity.score_single to always accept, i.e. drop "
                         "the state gate entirely (control experiment: is the "
                         "gate what blocks these tasks?)")
    ap.add_argument("--gate-ratio", type=float, default=1.0,
                    help="scale the gate's acceptance radii (chi2 joint radius "
                         "and per-dim sigma cap) by this factor; 1.0 = unchanged. "
                         "Tests how far outside the fitted support the blocked "
                         "states are")
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--lookahead", type=int, default=3,
                    help="evaluate best-K candidates with a follow-up step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-seed", type=int, default=None,
                    help="seed the ogbench env RNG per episode (episode i gets "
                         "default_rng(env_seed + i)) so the task stream is "
                         "reproducible across runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    scene = Scene.get(find_scene_config(args.scene), auto_load=False)
    model_cfgs = find_scene_models(args.scene)
    if args.loose_gate:
        from heca.data.entity import Entity

        # the graph gate calls ``score_prepared`` (memoised path), not
        # ``score_single``
        Entity.score_prepared = lambda self, sample, p, eps=1e-15: True  # type: ignore[assignment]
        Entity.score_single = lambda self, sample, up, eps=1e-15: True  # type: ignore[assignment]
        print("  [gate] Entity.score_prepared/score_single patched: every option "
              "is feasible", flush=True)
    elif args.gate_ratio != 1.0:
        import numpy as np

        from heca.data.entity import Entity, _chi_sqrt

        def _scaled(self, sample, p, eps=1e-15):  # noqa: ANN001
            sample = self.model_value(sample)
            pose, state = sample[:-1], int(sample[-1])
            pis = p["measurement"]["state"]["pis"]
            best_k, z, zd = self._best_component(pose, p, eps=eps)
            chi = _chi_sqrt(self.cfg.z_quantile_joint, len(pose)) * args.gate_ratio
            zd_cap = self._z_dim_sigma * args.gate_ratio
            if not (z <= chi and bool(np.all(zd <= zd_cap))):
                return False
            return bool(pis[best_k][state] > 1e-6)

        Entity.score_prepared = _scaled  # type: ignore[assignment]
        print(f"  [gate] acceptance radii scaled by {args.gate_ratio}", flush=True)
    for agent_cfg in model_cfgs:
        expert = ExpertModel.get(agent_cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
        _ = expert.conditions
    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode.BOTH)
    max_opts = scene.cfg.max_steps
    rng = np.random.RandomState(args.seed)

    # candidate evaluation: value of committing option i from state x
    def eval_option(x, y, i, snap, step_before):
        env_restore(scene, snap)
        scene.current_step = step_before
        graph.set_goal(y)
        graph.set_start(x)
        # ``Graph.select`` indexes the key list of the *last* export, so refresh
        # it for this state before selecting by index.
        graph.export()
        assert i < len(graph.export_keys), (i, len(graph.export_keys))
        a, s = graph.select(i)
        z, fb = ExpertModel.get(a).act(x, s)
        return z, fb

    def guided_chain(x0, y0, snap0, rng_local, use_lookahead: bool):
        x = x0
        n = 0
        trace = []
        for _ in range(max_opts):
            keys, dead = try_export(graph)
            if keys is None:
                return False, n, mismatch(x, y0), trace + ["DEAD_END"]
            snap = env_snapshot(scene)
            step_before = scene.current_step
            base = mismatch(x, y0)
            scored = []
            for i in range(len(keys)):
                z, fb = eval_option(x, y0, i, snap, step_before)
                if fb.terminal and fb.reward > 0.0:
                    return True, n + 1, 0.0, trace + [keys[i]]
                scored.append((base - mismatch(z, y0), i))
            # randomize among near-best to explore different orders
            scored.sort(reverse=True)
            best = scored[0][0]
            pool = [i for sc, i in scored if sc >= best - 1e-6] if best <= 0 else \
                   [i for sc, i in scored if sc >= best * 0.98]
            # lookahead: refine the top-K by their best follow-up
            if use_lookahead and len(scored) > 1:
                top = [i for _sc, i in scored[:args.lookahead]]
                refined = []
                for i in top:
                    z1, fb1 = eval_option(x, y0, i, snap, step_before)
                    graph.set_start(z1)
                    keys2, dead2 = try_export(graph)
                    if keys2 is None:
                        refined.append((base - mismatch(z1, y0), i))
                        continue
                    snap1 = env_snapshot(scene)
                    step1 = scene.current_step
                    best2 = None
                    for j in range(len(keys2)):
                        z2, fb2 = eval_option(z1, y0, j, snap1, step1)
                        sc2 = mismatch(z1, y0) - mismatch(z2, y0)
                        if best2 is None or sc2 > best2:
                            best2 = sc2
                    refined.append(((base - mismatch(z1, y0)) + (best2 or 0.0), i))
                refined.sort(reverse=True)
                pool = [i for _sc, i in refined[:max(1, args.lookahead // 2)]]
            pick = int(pool[rng_local.randint(len(pool))])
            z, fb = eval_option(x, y0, pick, snap, step_before)
            trace.append(keys[pick])
            n += 1
            x = z
            graph.set_start(x)
            if fb.terminal:
                return fb.reward > 0.0, n, mismatch(x, y0), trace
            if fb.truncated:
                return False, n, mismatch(x, y0), trace
        return False, n, mismatch(x, y0), trace

    results = []
    eps = ([int(e) for e in args.episode_list.split(",") if e.strip()]
           if args.episode_list else list(range(args.episodes)))
    for ep in eps:
        if args.env_seed is not None:
            scene.env.np_random = np.random.default_rng(args.env_seed + ep)
        (x0, _), (y0, _) = scene.sample_task()
        snap0 = env_snapshot(scene)
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)

        solved = False
        for _ in range(args.tries):
            env_restore(scene, snap0)
            scene.current_step = 0
            graph.set_goal(y0)
            graph.set_start(x0)
            ok, _ln, _dead = run_chain(
                x0, y0, graph, lambda n: int(rng.randint(n)), max_opts
            )
            if ok:
                solved = True
                break
        if solved:
            continue

        print(f"  [{args.scene}] ep {ep}: random oracle failed -> guided search",
              flush=True)
        best = None
        for r in range(args.restarts):
            env_restore(scene, snap0)
            scene.current_step = 0
            graph.set_goal(y0)
            graph.set_start(x0)
            rng_local = np.random.RandomState(1000 + r)
            ok, ln, mm, trace = guided_chain(
                x0, y0, snap0, rng_local,
                use_lookahead=(r % 2 == 1 and args.lookahead > 0),
            )
            print(f"      restart {r}: solved={ok} opts={ln} mismatch={mm:.4f}",
                  flush=True)
            if best is None or mm < best["mismatch"]:
                best = {"solved": ok, "opts": ln, "mismatch": mm,
                        "restart": r, "trace": trace[:10]}
            if ok:
                break
        results.append({"ep": ep, **best})

    res = {
        "scene": args.scene,
        "episodes": args.episodes,
        "tries": args.tries,
        "restarts": args.restarts,
        "env_seed": args.env_seed,
        "random_failures": len(results),
        "guided_solved": sum(1 for r in results if r["solved"]),
        "results": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[{args.scene}] random failures={len(results)}, "
          f"guided solved={res['guided_solved']}")
    for r in results:
        print(f"   ep {r['ep']}: solved={r['solved']} best_mismatch={r['mismatch']:.4f} "
              f"opts={r['opts']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
