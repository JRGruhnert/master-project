"""Regression check: optimized graph bookkeeping == pre-optimization behavior.

Two hot paths were rewritten for speed:

* ``Graph.feasible_keys`` -- now memoizes the pre-gate per agent and per step,
  prepares StepMix parameters once per fitted model, and reads subgoal values
  straight from the option's source nodes instead of copying the whole scene.
* ``ConditionEdges.build`` -- now scores every edge in one batched numpy pass.

Both are supposed to be bit-for-bit behavior preserving, so this script drives a
real episode loop and compares the optimized result against the original
implementation (re-implemented here verbatim) at every step:

* the ordered list of feasible option keys,
* the condition edge attributes (max abs difference).

It also reports how much time each implementation needs per step.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl \
        python scripts/diag_verify_gate.py --steps 200
"""

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.nodes.node import OptionNode
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def legacy_feasible_keys(graph: Graph) -> list[str]:
    """The pre-optimization ``feasible_keys``, verbatim."""
    values = []
    for key in graph.ns_option.keys:
        node = graph.ns_option.get_by_key(key)
        con = graph._option_conditions.get(key)
        if con is None or not graph._start_set:
            values.append(key)
            continue
        try:
            subgoal = graph.assemble_subgoal(node)
        except KeyError:
            continue
        ok = True
        for label in con.pre.models:
            up = con.pre.models[label].get_parameters().copy()
            try:
                value = graph.start.get(label).value
            except KeyError:
                ok = False
                break
            if not graph.entities[label].score_single(value, up):
                ok = False
                break
        if not ok:
            continue
        for label in con.post.models:
            up = con.post.models[label].get_parameters().copy()
            if not graph.entities[label].score_single(subgoal.get(label).value, up):
                ok = False
                break
        if ok:
            values.append(key)
    return values


def legacy_edge_attr(graph: Graph) -> torch.Tensor:
    """Recompute condition edge attributes the slow, per-edge way."""
    es = graph.es_condition
    attrs = []
    for s, d in es.edges:
        src = graph.ns_entity.idx_get(s)
        dst = graph.ns_entity.idx_get(d)
        attrs.append(
            es.stepmix_feat(src.data.feature, dst.data.feature, src.weight, src.n_states)
        )
    return torch.from_numpy(np.stack(attrs)).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="scene0")
    ap.add_argument("--smode", default="both")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    scene = Scene.get(find_scene_config(args.scene), auto_load=False)
    model_cfgs = find_scene_models(args.scene)
    for cfg in model_cfgs:
        expert = ExpertModel.get(cfg, auto_load=False)
        expert.use_gt(True)
        expert.load()
        expert.virtual()
    graph = Graph.generate(list(model_cfgs), smode=SubgoalMode(args.smode))

    (x, _), (y, _) = scene.sample_task()
    graph.set_goal(y)
    graph.set_start(x)

    key_mismatch = 0
    order_mismatch = 0
    empty_both = 0
    max_attr_diff = 0.0
    t_new = t_old = t_build_new = t_build_old = 0.0
    checked = 0

    for i in range(args.steps):
        # --- feasible key sets -------------------------------------------------
        t0 = time.perf_counter()
        try:
            new = graph.feasible_keys()
        except RuntimeError:
            new = None
        t1 = time.perf_counter()
        old = legacy_feasible_keys(graph)
        t2 = time.perf_counter()
        t_new += t1 - t0
        t_old += t2 - t1
        if new is None:
            empty_both += 1
            new = []
        if sorted(new) != sorted(old):
            key_mismatch += 1
            print(f"step {i}: KEY MISMATCH\n  new={sorted(new)}\n  old={sorted(old)}")
        elif new != old:
            order_mismatch += 1
        if not new:
            (x, _), (y, _) = scene.sample_task()
            graph.set_goal(y)
            graph.set_start(x)
            continue

        # --- condition edge attributes ---------------------------------------
        t3 = time.perf_counter()
        graph.es_condition.build(graph.ns_entity, graph.ns_entity)
        t4 = time.perf_counter()
        old_attr = legacy_edge_attr(graph)
        t5 = time.perf_counter()
        t_build_new += t4 - t3
        t_build_old += t5 - t4
        max_attr_diff = max(
            max_attr_diff, float((old_attr - graph.es_condition.edge_attr).abs().max())
        )
        checked += 1

        # --- advance one real option step -------------------------------------
        node = graph.ns_option.get_by_key(new[i % len(new)])
        assert isinstance(node, OptionNode)
        pose = graph.start.copy()
        for skey in node.sources.get("entity", set()):
            snode = graph.ns_entity.get_by_key(skey)
            pose.set(snode.entity, snode.data.copy())
        model = ExpertModel.get(node.model)
        z, fb = model.act(graph.start, pose)
        if fb.terminal or fb.truncated:
            (x, _), (y, _) = scene.sample_task()
            graph.set_goal(y)
            graph.set_start(z if fb.terminal else x)
        graph.set_start(z)
        if i % 20 == 0:
            graph.set_start(graph.start)  # exercise repeated set_start

    print(f"scene={args.scene} smode={args.smode} steps={args.steps}")
    print(f"  steps compared              : {checked}")
    print(f"  empty episodes (both)       : {empty_both}")
    print(f"  feasible-key set mismatches : {key_mismatch}")
    print(f"  feasible-key order mismatches: {order_mismatch}")
    print(f"  max |edge_attr diff|        : {max_attr_diff:.3e}")
    print(
        f"  feasible_keys ms/step       : new {1000 * t_new / args.steps:.3f} "
        f"| legacy {1000 * t_old / args.steps:.3f}"
    )
    print(
        f"  es_condition.build ms/step  : new {1000 * t_build_new / args.steps:.3f} "
        f"| legacy {1000 * t_build_old / args.steps:.3f}"
    )
    ok = key_mismatch == 0 and max_attr_diff == 0.0
    print("RESULT:", "IDENTICAL" if ok else "DIVERGENCE")


if __name__ == "__main__":
    main()
