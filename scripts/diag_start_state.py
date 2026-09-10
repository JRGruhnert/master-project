"""Audit the *task start* state: is the gate ever empty, and is the start pose
ever already "inside a container"?

The dead-ends reported by ``diag_deadends.py`` are created *during* a rollout (a
skill's post-condition teleports an entity into the box / onto the shelf / closes
the lid). This script checks the opposite direction: that the sampled initial
state is always a valid, gate-open, un-placed configuration.

    python scripts/diag_start_state.py --scene scene4 --tasks 200

Reports, per scene: the distribution of the number of feasible options at start,
how many tasks have an empty gate, and the level signature of every movable
entity in the start state (compared against the episode's own anchor poses).
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
from scripts.diag_deadend_conditions import level_signature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--env-seed", type=int, default=None,
                    help="optional: seed the env per task for reproducibility")
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

    n_feasible, zero, levels, box_ste = [], 0, collections.Counter(), collections.Counter()
    for t in range(args.tasks):
        if args.env_seed is not None:
            scene.env.np_random = np.random.default_rng(args.env_seed + t)
        (x0, _), (y0, _) = scene.sample_task()
        graph.set_goal(y0)
        graph.set_start(x0)
        try:
            keys = graph.feasible_keys()
        except RuntimeError:
            keys = []
            zero += 1
        n_feasible.append(len(keys))
        for label, d in level_signature(x0).items():
            levels[(label, d["level"])] += 1
        box_ste[int(np.asarray(x0.get("box0").value).ravel()[-1])] += 1

    res = {
        "scene": args.scene, "tasks": args.tasks, "env_seed": args.env_seed,
        "n_options": len(graph.ns_option.keys),
        "feasible_at_start_min": int(min(n_feasible)),
        "feasible_at_start_mean": round(float(np.mean(n_feasible)), 2),
        "feasible_at_start_max": int(max(n_feasible)),
        "tasks_with_empty_gate": zero,
        "start_level_hist": {f"{k[0]}@{k[1]}": v for k, v in levels.most_common()},
        "start_box0_state_hist": {str(k): v for k, v in sorted(box_ste.items())},
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[{args.scene}] {args.tasks} tasks: feasible@start "
          f"{res['feasible_at_start_min']}..{res['feasible_at_start_max']} "
          f"(mean {res['feasible_at_start_mean']}), empty gate in "
          f"{zero}/{args.tasks} tasks")
    print(f"   start levels: {res['start_level_hist']}")
    print(f"   box0 state at start: {res['start_box0_state_hist']}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
