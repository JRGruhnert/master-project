"""Probe how an option's subgoal is resolved entity-by-entity.

``Graph.assemble_subgoal`` walks ``option.sources["entity"]`` (a *set*) and lets
the last visited node win for each entity. Several value nodes can share an
entity (the START / GOAL / SAMPLE variants created in ``set_postcon``), so the
resolved subgoal can depend on set iteration order. This probe reports, for
every option, which entities have several candidate sources and which candidate
each of them would supply.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl \
        python scripts/diag_subgoal_sources.py --scene scene0 --smode both
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np

from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.graphs.nodes.node import OptionNode, ValueNode
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="scene0")
    ap.add_argument("--smode", default="both")
    ap.add_argument("--max-options", type=int, default=12)
    args = ap.parse_args()

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

    multi = 0
    for key in graph.ns_option.keys:
        node = graph.ns_option.get_by_key(key)
        assert isinstance(node, OptionNode)
        by_entity: dict[str, list[str]] = defaultdict(list)
        for skey in node.sources.get("entity", set()):
            if graph.ns_entity.has_key(skey):
                by_entity[graph.ns_entity.get_by_key(skey).entity].append(skey)
        dup = {e: ks for e, ks in by_entity.items() if len(ks) > 1}
        if not dup:
            continue
        multi += 1
        if multi > args.max_options:
            continue
        print(f"\noption {key}: {len(dup)} entities with multiple sources")
        sub = graph.assemble_subgoal(node)
        for entity, keys in dup.items():
            kinds = []
            for k in keys:
                n = graph.ns_entity.get_by_key(k)
                assert isinstance(n, ValueNode)
                kinds.append(f"{n.vmode.value}:{k}")
            resolved = np.asarray(sub.get(entity).value, dtype=float)
            start_v = np.asarray(graph.start.get(entity).value, dtype=float)
            goal_v = np.asarray(graph.goal.get(entity).value, dtype=float)
            which = "start" if np.allclose(resolved, start_v) else (
                "goal" if np.allclose(resolved, goal_v) else "other"
            )
            print(f"  {entity:<24} sources={kinds}")
            print(
                f"      resolved-> {which:<6} "
                f"|start-res|={np.linalg.norm(resolved - start_v):.4f} "
                f"|goal-res|={np.linalg.norm(resolved - goal_v):.4f}"
            )

    print(
        f"\n{multi} of {len(graph.ns_option.keys)} options have at least one "
        "entity fed by several value nodes."
    )


if __name__ == "__main__":
    main()
