"""Targeted deep oracle retries for episodes the base run could not solve.

Replays the exact same episode stream as ``diag_scene_reachability.py`` with the
same (scene, seed, episodes, tries) so the base phase is bit-identical, then
gives the *unsolved* episodes many more random-chain retries. The env RNG state
is snapshotted after the base tries and restored after the deep tries, so
subsequent episodes are sampled exactly as in the original run.

Usage:
    python scripts/diag_deep_retry.py --scene scene10 --episodes 60 \
        --tries 60 --deep-tries 500 --out data/diag/deep_scene10.json
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


def rng_state(env):
    return env.np_random.bit_generator.state


def rng_set(env, state):
    env.np_random.bit_generator.state = state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--tries", type=int, default=60)
    ap.add_argument("--deep-tries", type=int, default=500)
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
    max_opts = scene.cfg.max_steps
    rng = np.random.RandomState(args.seed)

    base_solved = 0
    deep_solved = 0
    flips = []
    rows = []

    for ep in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()
        snap0 = env_snapshot(scene)
        scene.current_step = 0
        graph.set_goal(y0)
        graph.set_start(x0)
        keys0, dead0 = try_export(graph)

        def attempt(n_tries):
            for _ in range(n_tries):
                env_restore(scene, snap0)
                scene.current_step = 0
                graph.set_goal(y0)
                graph.set_start(x0)
                ok, ln, _dead = run_chain(
                    x0, y0, graph, lambda n: int(rng.randint(n)), max_opts
                )
                if ok:
                    return True, ln
            return False, None

        solved = False
        ln = None
        used = 0
        if keys0 is not None:
            solved, ln = attempt(args.tries)
            used = args.tries

        deep = solved
        deep_len = ln if solved else None
        if not solved and keys0 is not None:
            # Deep retries: keep the env RNG frozen for the *next* episode so
            # the base stream stays aligned with the original run.
            st_after_base = rng_state(scene.env)
            extra = max(args.deep_tries - args.tries, 0)
            deep, deep_len = attempt(extra)
            used += extra
            rng_set(scene.env, st_after_base)
            if deep:
                flips.append({"ep": ep, "deep_chain_len": deep_len})

        base_solved += int(solved)
        deep_solved += int(deep)
        rows.append(
            {"ep": ep, "base_solved": solved, "deep_solved": deep,
             "deep_len": deep_len, "dead_end_at_start": dead0 is not None}
        )
        print(f"  [{args.scene}] ep {ep+1}/{args.episodes} "
              f"base={base_solved} deep={deep_solved}"
              + ("  <-- flipped" if (deep and not solved) else ""), flush=True)

    res = {
        "scene": args.scene,
        "episodes": args.episodes,
        "base_tries": args.tries,
        "deep_tries": args.deep_tries,
        "seed": args.seed,
        "base_solved": base_solved,
        "base_pct": 100.0 * base_solved / args.episodes,
        "deep_solved": deep_solved,
        "deep_pct": 100.0 * deep_solved / args.episodes,
        "flips": flips,
        "still_unsolved": [
            r["ep"] for r in rows if not r["deep_solved"]
        ],
        "elapsed_s": round(time.time() - t0, 1),
        "per_episode": rows,
    }
    print(f"[{args.scene}] base {base_solved}/{args.episodes} "
          f"({res['base_pct']:.1f}%) -> deep {deep_solved}/{args.episodes} "
          f"({res['deep_pct']:.1f}%)  flips={len(flips)} "
          f"still_unsolved={res['still_unsolved']}  ({res['elapsed_s']}s)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
