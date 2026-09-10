"""Profile the per-step graph bookkeeping that dominates wall-clock time.

The RL loop for one option step is

    data = graph.export()      # feasible_keys + compaction  (the hot part)
    logits = network(data)
    a, s = graph.select(i)
    z = expert.act(x, s)
    graph.set_start(z)         # update_nodes + rebuild (es_condition.build)

This script builds the same graph as the training run (scene0, both subgoal
modes, virtual GT experts) and reports where the per-step milliseconds go, both
per top-level call and per function (cProfile). Use it before/after touching the
graph code; the totals are what a training update actually pays.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl \
        python scripts/diag_profile_step.py --steps 60
"""

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
# Unconditional: a stale/relative copy of the repo root may already sit further
# back in sys.path, in which case site-packages' ``conf`` (ogbench) wins.
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

import conf.networks
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.heca_gnn.network import Network
from heca.misc import hardware
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="scene0")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--smode", default="both")
    ap.add_argument("--network", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene = Scene.get(find_scene_config(args.scene), auto_load=False)
    model_cfgs = find_scene_models(args.scene)
    for agent_cfg in model_cfgs:
        expert = ExpertModel.get(agent_cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()

    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode(args.smode))
    net_cfg = getattr(conf.networks, args.network)
    network = Network.get(net_cfg)
    network.eval()

    (x, _), (y, _) = scene.sample_task()
    graph.set_goal(y)
    graph.set_start(x)

    # Warm up (first call builds caches inside torch / numpy paths).
    for _ in range(3):
        data = graph.export()
        with torch.inference_mode():
            network(data)

    timings: dict[str, list[float]] = {}

    def step():
        t0 = time.perf_counter()
        data = graph.export()
        t1 = time.perf_counter()
        with torch.inference_mode():
            network.actor(data)
        t2 = time.perf_counter()
        _, s = graph.select(0)
        expert = ExpertModel.get(graph.ns_option.items[0].model)
        z, _ = expert.act(graph.start, s)
        t3 = time.perf_counter()
        graph.set_start(z)
        t4 = time.perf_counter()
        for name, dt in (
            ("graph.export", t1 - t0),
            ("network.forward", t2 - t1),
            ("expert.act", t3 - t2),
            ("set_start+rebuild", t4 - t3),
            ("TOTAL", t4 - t0),
        ):
            timings.setdefault(name, []).append(dt)

    # Wall clock first, WITHOUT cProfile: the profiler's per-call overhead
    # inflates these numbers by more than an order of magnitude.
    for _ in range(args.steps):
        try:
            step()
        except RuntimeError:
            # feasible_keys() empty -> episode is a dead end, resample.
            (x, _), (y, _) = scene.sample_task()
            graph.set_goal(y)
            graph.set_start(x)

    print(f"=== per-step wall clock (mean over {args.steps} steps) ===")
    total = np.mean(timings["TOTAL"]) * 1000.0
    for name, vals in timings.items():
        ms = np.mean(vals) * 1000.0
        share = 100.0 * ms / total
        print(f"  {name:<20} {ms:8.3f} ms  ({share:5.1f}%)")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(max(5, args.steps // 8)):
        try:
            step()
        except RuntimeError:
            pass
    profiler.disable()

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("tottime")
    stats.print_stats(args.top)
    print("\n=== cProfile (counts are per profiled step here) ===")
    for line in buf.getvalue().splitlines():
        print(line)


if __name__ == "__main__":
    main()
